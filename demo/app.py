import argparse
import os

import hydra
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from utils import load_ply
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from pc_sam.scene_inference import TiledSceneSAM
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="pretrained/model.safetensors")
parser.add_argument("--pointcloud", type=str, default="scene.ply")
parser.add_argument(
    "--num_points",
    type=int,
    default=200000,
    help="number of points sent to the browser for DISPLAY (not a model limit; "
    "the full cloud is always encoded via tiling)",
)
parser.add_argument(
    "--max_core_points",
    type=int,
    default=150000,
    help="max points per encoded tile (kd-split cap); smaller = finer/safer VRAM",
)
parser.add_argument("--overlap", type=float, default=0.1, help="tile halo fraction")
parser.add_argument("--config", type=str, default="large", help="path to config file")
parser.add_argument("--config_dir", type=str, default="../configs")
args = parser.parse_args()

output_dir = "results"

# ---------------------------------------------------------------------------- #
# Setup model
# ---------------------------------------------------------------------------- #
with hydra.initialize(args.config_dir, version_base=None):
    cfg = hydra.compose(config_name=args.config)
    OmegaConf.resolve(cfg)

model = hydra.utils.instantiate(cfg.model)
model.apply(replace_with_fused_layernorm)
load_model(model, args.checkpoint)
model.eval().cuda()

# ---------------------------------------------------------------------------- #
# Load the FULL-resolution cloud and encode it as cached spatial tiles.
# This replaces the old "globally subsample then encode" path: every point is
# encoded (no points discarded), within bounded VRAM, and each click reuses the
# cached per-tile embedding (see pc_sam/scene_inference.py).
# ---------------------------------------------------------------------------- #
def _resolve_cloud(p):
    if os.path.exists(p):
        return p
    return os.path.join(os.path.dirname(__file__), "static", "models", p)

src = _resolve_cloud(args.pointcloud)
obj_path = os.path.basename(args.pointcloud)
print(f"[load] {src}", flush=True)
points = load_ply(src)
xyz_world = points[:, :3].astype(np.float64)
rgb01 = (points[:, 3:6] / 255).astype(np.float32)
print(f"[load] {xyz_world.shape[0]} points; encoding tiles ...", flush=True)

segmenter = TiledSceneSAM(
    model,
    max_core_points=args.max_core_points,
    overlap=args.overlap,
)
segmenter.encode_scene(xyz_world, rgb01)
segmenter.new_session()

# Display subsample: the browser cannot render millions of points. We render a
# subset, but segmentation runs on the full cloud and is mapped back to it.
N = xyz_world.shape[0]
if N > args.num_points:
    disp_idx = np.sort(np.random.choice(N, args.num_points, replace=False))
else:
    disp_idx = np.arange(N)
disp_world = xyz_world[disp_idx]
disp_rgb = rgb01[disp_idx]
_dshift = disp_world.mean(0)
_dscale = max(float(np.linalg.norm(disp_world - _dshift, axis=1).max()), 1e-8)
disp_norm = ((disp_world - _dshift) / _dscale).astype(np.float32)  # browser frame
disp_tree = cKDTree(disp_norm)  # map browser clicks (normalized) -> displayed point
print(f"[display] sending {len(disp_idx)} of {N} points to the browser", flush=True)

# session state
cur_full_mask = None          # latest [N] bool mask
saved_masks = []              # list of [N] bool masks (one per confirmed object)

# Flask Backend
app = Flask(__name__, static_folder="static")
CORS(app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/static/<path:path>")
def static_server(path):
    return app.send_static_file(path)


@app.route("/mesh/<path:path>")
def mesh_server(path):
    return app.send_static_file(f"models/{path}")


@app.route("/pointcloud/<path:path>")
def pointcloud_server(path):
    # Serve the (normalized) display subsample. Segmentation still runs full-res.
    return jsonify({"xyz": disp_norm.flatten().tolist(), "rgb": disp_rgb.flatten().tolist()})


@app.route("/clear", methods=["POST"])
def clear():
    global cur_full_mask
    segmenter.new_session()
    cur_full_mask = None
    return jsonify({"status": "cleared"})


@app.route("/next", methods=["POST"])
def next():
    global cur_full_mask
    if cur_full_mask is not None:
        saved_masks.append(cur_full_mask.copy())
    segmenter.new_session()
    cur_full_mask = None
    return jsonify({"status": "cleared"})


@app.route("/save", methods=["POST"])
def save():
    os.makedirs(output_dir, exist_ok=True)
    masks = list(saved_masks)
    if cur_full_mask is not None:
        masks.append(cur_full_mask)
    name = obj_path.split(".")[0]
    # full-resolution export: per-point xyz/rgb + one boolean mask per object
    np.save(
        f"{output_dir}/{name}.npy",
        {"xyz": xyz_world, "rgb": rgb01, "masks": np.stack(masks) if masks else np.zeros((0, N), bool)},
    )
    return jsonify({"status": "saved", "num_objects": len(masks)})


@app.route("/segment", methods=["POST"])
def segment():
    global cur_full_mask
    request_data = request.get_json()
    prompt_label = int(request_data["prompt_label"])

    # Preferred: the browser sends the picked display-point INDEX (exact), so we
    # recover its world coordinate directly. Fallback: snap a sent coordinate to
    # the nearest displayed point. Either way we get a real world point, which
    # routes the prompt to the correct cached tile.
    if request_data.get("prompt_index") is not None:
        world = disp_world[int(request_data["prompt_index"])]
    else:
        prompt_point = np.array(request_data["prompt_point"], dtype=np.float64).reshape(3)
        _, nn = disp_tree.query(prompt_point)
        world = disp_world[nn]

    full_mask = segmenter.add_prompt(world, prompt_label)  # [N] bool, full-res
    cur_full_mask = full_mask

    # Return the mask over the displayed points (browser only knows those).
    seg_disp = full_mask[disp_idx]
    return jsonify({"seg": seg_disp.tolist()})


if __name__ == "__main__":
    # use_reloader=False so the heavy scene encoding does not run twice.
    app.run(host=f"{args.host}", port=f"{args.port}", debug=True, use_reloader=False)
