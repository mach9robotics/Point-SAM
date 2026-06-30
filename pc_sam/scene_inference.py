"""Tiled, cache-based scene inference for Point-SAM on large point clouds.

Point-SAM is a *crop/object-level* promptable model: its encoder tokenizes a
cloud into at most ~2048 FPS patches, so feeding a multi-million-point scene at
once is both (a) an OOM — the encoder/decoder each build an O(num_patches * N)
`cdist` matrix (~26 GB at N=3.2M, num_patches=2048) — and (b) far too coarse.

This module follows the design validated by SAM (ICCV'23) and AGILE3D (ICLR'24)
and used by Point-SAM's own paper (3 m blocks, 1.5 m stride): split the scene
into spatially-compact tiles small enough to fit VRAM, **encode each tile once**
and **cache** its embedding, then run only the lightweight prompt decoder per
click against the relevant tile's cached state. The decoder interpolates back to
*all* points in the tile, so the output is full-resolution — no points dropped
(contrast the naive demo, which globally subsampled the cloud before encoding).

Memory: each tile is capped at `max_core_points` (kd-split guarantees it), so the
native `cdist` always fits; only one tile is resident on GPU during a decode. The
per-tile cache lives on CPU RAM (or disk via `cache_dir`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .model.pc_sam import PointCloudSAM
from .model.mask_decoder import AuxInputs
from .model.common import compute_interp_weights, repeat_interleave


@dataclass
class TileCache:
    """Cached, reusable encoder state for one spatial tile (SAM's 'image embedding')."""

    tile_id: int
    encoded_idx: np.ndarray          # [n] global indices of points encoded in this tile
    n_core: int                      # first n_core entries of encoded_idx are this tile's "own" points
    shift: np.ndarray                # [3] world->local: local = (world - shift) / scale
    scale: float
    # cached tensors (kept on CPU; moved to GPU on demand). All have batch dim 1.
    coords: torch.Tensor             # [1, n, 3] normalized to unit sphere
    features: torch.Tensor           # [1, n, 3]
    pc_embeddings: torch.Tensor      # [1, L, D]
    centers: torch.Tensor            # [1, L, 3]
    knn_idx: torch.Tensor            # [1, L, K]
    pc_pe: torch.Tensor              # [1, L, D]
    interp_index: torch.Tensor       # [1, n, 3]
    interp_weight: torch.Tensor      # [1, n, 3]


def _kd_split(idx: np.ndarray, coords: np.ndarray, max_points: int) -> list[np.ndarray]:
    """Recursively median-split point indices along the longest axis until every
    leaf has <= max_points points. Produces spatially-compact, balanced tiles."""
    if len(idx) <= max_points:
        return [idx]
    pts = coords[idx]
    extent = pts.max(0) - pts.min(0)
    axis = int(np.argmax(extent))
    med = np.median(pts[:, axis])
    left_mask = pts[:, axis] <= med
    # Degenerate case (e.g. many identical coords): fall back to an even split.
    if left_mask.all() or (~left_mask).all():
        half = len(idx) // 2
        order = np.argsort(pts[:, axis], kind="stable")
        left, right = idx[order[:half]], idx[order[half:]]
    else:
        left, right = idx[left_mask], idx[~left_mask]
    return _kd_split(left, coords, max_points) + _kd_split(right, coords, max_points)


class TiledSceneSAM:
    """Encode a large point cloud as cached spatial tiles; segment from cache per prompt.

    Typical use:
        seg = TiledSceneSAM(model)
        seg.encode_scene(xyz_world, rgb01)          # one-time, builds + caches all tiles
        mask = seg.segment(prompt_xyz, prompt_labels)   # cheap, per click -> [N] bool
    """

    def __init__(
        self,
        model: PointCloudSAM,
        max_core_points: int = 150_000,
        overlap: float = 0.1,
        max_encoded_points: int = 300_000,
        num_groups: int = 2048,
        group_size: int = 256,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        self.model = model.eval()
        self.max_core_points = max_core_points
        self.overlap = overlap
        self.max_encoded_points = max(max_encoded_points, max_core_points)
        self.num_groups = num_groups
        self.group_size = group_size
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self._xyz: Optional[np.ndarray] = None      # [N, 3] world coords
        self._rgb: Optional[np.ndarray] = None       # [N, 3] in [0, 1]
        self._tiles: list[TileCache] = []
        self._point_to_tile: Optional[np.ndarray] = None  # [N] core-tile id per point
        self._kdtree = None                          # scipy cKDTree over world coords

    # ------------------------------------------------------------------ #
    # Encoding (one-time, expensive part — cached)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_scene(self, xyz_world: np.ndarray, rgb01: np.ndarray, verbose: bool = True):
        """Tile the scene, encode each tile once, and cache the embeddings."""
        from scipy.spatial import cKDTree

        xyz_world = np.ascontiguousarray(xyz_world, dtype=np.float64)
        rgb01 = np.ascontiguousarray(rgb01, dtype=np.float32)
        assert xyz_world.shape[0] == rgb01.shape[0]
        self._xyz, self._rgb = xyz_world, rgb01
        N = xyz_world.shape[0]

        core_tiles = _kd_split(np.arange(N), xyz_world, self.max_core_points)
        self._point_to_tile = np.empty(N, dtype=np.int64)
        for tid, core in enumerate(core_tiles):
            self._point_to_tile[core] = tid

        # configure encoder grouping for the tile size regime (mirrors eval_kitti.py)
        self.model.pc_encoder.patch_embed.grouper.num_groups = self.num_groups
        self.model.pc_encoder.patch_embed.grouper.group_size = self.group_size

        self._tiles = []
        for tid, core in enumerate(core_tiles):
            encoded_idx = self._add_halo(core, xyz_world)
            tile = self._encode_tile(tid, core, encoded_idx, xyz_world, rgb01)
            self._tiles.append(tile)
            if verbose:
                print(f"  tile {tid:>2}: core={len(core):>7} encoded={len(encoded_idx):>7} "
                      f"| L={tile.pc_embeddings.shape[1]}", flush=True)

        self._kdtree = cKDTree(xyz_world)
        if verbose:
            print(f"encoded {len(self._tiles)} tiles over {N} points", flush=True)
        return self

    def _add_halo(self, core_idx: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Expand a tile's core AABB by `overlap` and include neighbouring points,
        so objects on tile borders are not cut. Core points always come first."""
        if self.overlap <= 0:
            return core_idx
        lo, hi = xyz[core_idx].min(0), xyz[core_idx].max(0)
        margin = (hi - lo) * self.overlap
        lo, hi = lo - margin, hi + margin
        in_box = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        in_box[core_idx] = False  # exclude core (added explicitly, kept first)
        halo_idx = np.nonzero(in_box)[0]
        budget = self.max_encoded_points - len(core_idx)
        if budget <= 0:
            return core_idx
        if len(halo_idx) > budget:
            halo_idx = np.random.choice(halo_idx, budget, replace=False)
        return np.concatenate([core_idx, halo_idx])

    @torch.no_grad()
    def _encode_tile(self, tid, core_idx, encoded_idx, xyz, rgb) -> TileCache:
        pts = xyz[encoded_idx]
        shift = pts.mean(0)
        scale = float(np.linalg.norm(pts - shift, axis=1).max())
        scale = max(scale, 1e-8)
        coords_local = ((pts - shift) / scale).astype(np.float32)

        coords = torch.from_numpy(coords_local).to(self.device).unsqueeze(0)
        feats = torch.from_numpy(rgb[encoded_idx]).to(self.device).unsqueeze(0)

        with torch.autocast(self.device, dtype=self.dtype):
            pc_embeddings, patches = self.model.pc_encoder(coords, feats)
            centers = patches["centers"]
            knn_idx = patches["knn_idx"]
            pc_pe = self.model.point_encoder.pe_layer(centers)
            # Precompute the upsampling weights now (the decoder would otherwise build
            # an O(N*L) cdist lazily on the first prompt); cache them so every click is cheap.
            interp_index, interp_weight = compute_interp_weights(coords, centers)

        def cpu(t):
            return t.detach().to("cpu")

        tile = TileCache(
            tile_id=tid,
            encoded_idx=encoded_idx,
            n_core=len(core_idx),
            shift=shift,
            scale=scale,
            coords=cpu(coords),
            features=cpu(feats),
            pc_embeddings=cpu(pc_embeddings),
            centers=cpu(centers),
            knn_idx=cpu(knn_idx),
            pc_pe=cpu(pc_pe),
            interp_index=cpu(interp_index),
            interp_weight=cpu(interp_weight),
        )
        if self.cache_dir:
            torch.save(tile, os.path.join(self.cache_dir, f"tile_{tid:04d}.pt"))
        del coords, feats, pc_embeddings, patches, centers, knn_idx, pc_pe, interp_index, interp_weight
        torch.cuda.empty_cache()
        return tile

    # ------------------------------------------------------------------ #
    # Decoding (cheap, per-prompt — runs against one cached tile)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _decode_tile(self, tile: TileCache, prompt_coords, prompt_labels,
                     prompt_mask=None, multimask_output=True):
        """Run only the prompt encoder + mask decoder against a cached tile."""
        m = self.model
        dev = self.device
        coords = tile.coords.to(dev)
        features = tile.features.to(dev)
        centers = tile.centers.to(dev)
        knn_idx = tile.knn_idx.to(dev)
        pc_embeddings = tile.pc_embeddings.to(dev)
        pc_pe = tile.pc_pe.to(dev)
        aux = AuxInputs(
            coords=coords, features=features, centers=centers,
            interp_index=tile.interp_index.to(dev),
            interp_weight=tile.interp_weight.to(dev),
        )
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(dev)
        with torch.autocast(dev, dtype=self.dtype):
            sparse = m.point_encoder(prompt_coords, prompt_labels)
            dense = m.mask_encoder(prompt_mask, coords, centers, knn_idx)
            dense = repeat_interleave(dense, sparse.shape[0] // dense.shape[0], 0)
            masks, iou = m.mask_decoder(
                pc_embeddings, pc_pe, sparse, dense, aux_inputs=aux,
                multimask_output=multimask_output,
            )
        return masks, iou

    def _route(self, prompt_xyz_world: np.ndarray) -> int:
        """Return the core-tile id for a world-space prompt (the first positive click)."""
        _, nn = self._kdtree.query(prompt_xyz_world[0])
        return int(self._point_to_tile[nn])

    @torch.no_grad()
    def segment(
        self,
        prompt_xyz_world: np.ndarray,   # [P, 3] world coords (P clicks)
        prompt_labels: np.ndarray,      # [P] 1=positive, 0=negative
        prompt_mask: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ):
        """Segment from cached tiles. Returns a full-resolution [N] bool mask
        (and, if return_logits, the per-encoded-point logits + the tile used)."""
        assert self._tiles, "call encode_scene() first"
        prompt_xyz_world = np.asarray(prompt_xyz_world, dtype=np.float64).reshape(-1, 3)
        prompt_labels = np.asarray(prompt_labels).reshape(-1)

        tid = self._route(prompt_xyz_world)
        tile = self._tiles[tid]

        # map world prompts into this tile's local normalized frame. Clamp to the
        # unit cube so a click slightly outside the tile won't trip the encoder's
        # [-1, 1] positional-encoding assertion.
        local = (prompt_xyz_world - tile.shift) / tile.scale
        local = np.clip(local, -1.0, 1.0)
        pc = torch.from_numpy(local.astype(np.float32)).to(self.device).unsqueeze(0)
        pl = torch.from_numpy(prompt_labels.astype(np.int64)).to(self.device).unsqueeze(0)

        masks, iou = self._decode_tile(tile, pc, pl, prompt_mask, multimask_output)
        best = int(iou[0].argmax())
        logits = masks[0, best]                     # [n_encoded]
        positive = (logits > 0).detach().cpu().numpy()

        full_mask = np.zeros(self._xyz.shape[0], dtype=bool)
        full_mask[tile.encoded_idx] = positive
        if return_logits:
            return full_mask, logits.detach().cpu(), tid
        return full_mask

    # ------------------------------------------------------------------ #
    # Stateful interactive session (accumulate clicks for one object, with
    # SAM-style iterative mask-logit feedback). Used by the demo backend.
    # ------------------------------------------------------------------ #
    def new_session(self):
        self._sess = dict(prompts=[], labels=[], prompt_mask=None)

    @torch.no_grad()
    def add_prompt(self, world_xyz, label: int):
        """Add one click (world coords, 1=pos/0=neg) to the current object and
        return the updated full-resolution [N] bool mask."""
        s = getattr(self, "_sess", None)
        if s is None:
            self.new_session(); s = self._sess
        s["prompts"].append([float(c) for c in world_xyz])
        s["labels"].append(int(label))
        multimask = len(s["prompts"]) == 1  # multimask on the first click only
        full_mask, logits, _ = self.segment(
            np.array(s["prompts"]), np.array(s["labels"]),
            prompt_mask=s["prompt_mask"], multimask_output=multimask,
            return_logits=True,
        )
        s["prompt_mask"] = logits[None]  # feed best-mask logits back next click
        return full_mask

    # ------------------------------------------------------------------ #
    @property
    def num_tiles(self) -> int:
        return len(self._tiles)
