import hydra
from omegaconf import OmegaConf
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import load_ply
import numpy as np
import torch
import os
import numpy as np
import argparse
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from pc_sam.model.loss import compute_iou
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="pretrained/model.safetensors")
parser.add_argument("--pointcloud", type=str, default="scene.ply")
parser.add_argument(
    "--num_points",
    type=int,
    default=100000,
    help="randomly subsample the cloud to at most this many points (GPU/browser limit)",
)
parser.add_argument(
    "--config", type=str, default="large", help="path to config file"
)
parser.add_argument("--config_dir", type=str, default="../configs")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="./pretrained/ours/mixture_10k_giant/model.safetensors",
)
args = parser.parse_args()

# PCSAM variables
pc_xyz, pc_rgb = None, None
prompts, labels = [], []
prompt_mask = None
obj_path = None
output_dir = "results"
segment_mask = None
masks = []

# Flask Backend
app = Flask(__name__, static_folder="static")
CORS(
    app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin"
)

# change "./pretrained/model.safetensors" to the path of the checkpoint

# ---------------------------------------------------------------------------- #
# Load configuration
# ---------------------------------------------------------------------------- #
with hydra.initialize(args.config_dir, version_base=None):
    cfg = hydra.compose(config_name=args.config)
    OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg))


# ---------------------------------------------------------------------------- #
# Setup model
# ---------------------------------------------------------------------------- #
model = hydra.utils.instantiate(cfg.model)
model.apply(replace_with_fused_layernorm)

# ---------------------------------------------------------------------------- #
# Load pre-trained model
# ---------------------------------------------------------------------------- #
load_model(model, args.checkpoint)
model.eval().cuda()
sam = model


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/static/<path:path>")
def static_server(path):
    return app.send_static_file(path)


@app.route("/mesh/<path:path>")
def mesh_server(path):
    # path = f"/home/yuchen/workspace/annotator_3d/src/static/models/{path}"
    # print(path)
    # path = "/home/yuchen/workspace/annotator_3d/src/static/models/Rhino/White Rhino.obj"
    path = f"models/{path}"
    print(path)
    return app.send_static_file(path)


@app.route("/sampled_pointcloud", methods=["POST"])
def sampled_pc():
    request_data = request.get_json()
    points = request_data["points"].values()
    points = np.array(list(points)).reshape(-1, 3)
    colors = request_data["colors"].values()
    colors = np.array(list(colors)).reshape(-1, 3)

    global pc_xyz, pc_rgb
    pc_xyz, pc_rgb = (
        torch.from_numpy(points).cuda().float(),
        torch.from_numpy(colors).cuda().float(),
    )
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)

    response = "success"
    return jsonify({"response": response})


@app.route("/pointcloud/<path:path>")
def pointcloud_server(path):
    global obj_path, pc_xyz, pc_rgb

    # Resolve the cloud path: use --pointcloud as given if it exists (absolute or
    # relative to cwd), otherwise fall back to the bundled demo/static/models dir.
    src = args.pointcloud
    if not os.path.exists(src):
        src = os.path.join(os.path.dirname(__file__), "static", "models", args.pointcloud)
    obj_path = os.path.basename(args.pointcloud)

    points = load_ply(src)
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255

    # Subsample large clouds (e.g. multi-million-point LiDAR) to keep the encoder
    # within GPU memory and the browser payload manageable.
    if xyz.shape[0] > args.num_points:
        indices = np.random.choice(xyz.shape[0], args.num_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
    print(f"[pointcloud] {src}: serving {xyz.shape[0]} points")

    # normalize to a unit sphere centered at the origin
    shift = xyz.mean(0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    xyz = (xyz - shift) / scale

    # set pcsam variables (kept aligned with what we send to the browser)
    pc_xyz = torch.from_numpy(xyz).cuda().float().unsqueeze(0)
    pc_rgb = torch.from_numpy(rgb).cuda().float().unsqueeze(0)

    return jsonify({"xyz": xyz.flatten().tolist(), "rgb": rgb.flatten().tolist()})


@app.route("/clear", methods=["POST"])
def clear():
    global prompts, labels, prompt_mask, segment_mask
    prompts, labels = [], []
    prompt_mask = None
    segment_mask = None
    return jsonify({"status": "cleared"})


@app.route("/next", methods=["POST"])
def next():
    global prompts, labels, segment_mask, masks, prompt_mask
    masks.append(segment_mask.cpu().numpy())
    prompts, labels = [], []
    prompt_mask = None
    return jsonify({"status": "cleared"})


@app.route("/save", methods=["POST"])
def save():
    os.makedirs(output_dir, exist_ok=True)
    global pc_xyz, pc_rgb, segment_mask, obj_path, masks
    xyz = pc_xyz[0].cpu().numpy()
    rgb = pc_rgb[0].cpu().numpy()
    masks = np.stack(masks)
    obj_path = obj_path.split(".")[0]
    np.save(f"{output_dir}/{obj_path}.npy", {"xyz": xyz, "rgb": rgb, "mask": masks})
    global prompts, labels, prompt_mask
    prompts, labels = [], []
    prompt_mask = None
    segment_mask = None
    return jsonify({"status": "saved"})


@app.route("/segment", methods=["POST"])
def segment():
    request_data = request.get_json()
    prompt_point = request_data["prompt_point"]
    prompt_label = request_data["prompt_label"]

    # append prompt
    global prompts, labels, prompt_mask
    prompts.append(prompt_point)
    labels.append(prompt_label)

    prompt_points = torch.from_numpy(np.array(prompts)).cuda().float()[None, ...]
    prompt_labels = torch.from_numpy(np.array(labels)).cuda()[None, ...]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, iou_preds = sam.predict_masks(
            pc_xyz,
            pc_rgb,
            prompt_points,
            prompt_labels,
            prompt_mask,
            multimask_output=prompt_mask is None,
        )
    best = torch.argmax(iou_preds[0])
    prompt_mask = masks[0][best][None, ...]  # raw-logit mask fed back on next click
    global segment_mask
    segment_mask = return_mask = masks[0][best] > 0
    return jsonify({"seg": return_mask.cpu().numpy().tolist()})


if __name__ == "__main__":
    app.run(host=f"{args.host}", port=f"{args.port}", debug=True)
