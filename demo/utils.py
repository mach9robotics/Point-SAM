import numpy as np
import trimesh


def load_ply(filename):
    """Load a PLY point cloud and return an [N, 6] array of XYZ + RGB (0-255).

    Uses trimesh so it handles both ASCII and binary PLY, and tolerates extra
    vertex properties (e.g. KITTI360 semantic/instance/visible/confidence).
    """
    geom = trimesh.load(filename, process=False)
    xyz = np.asarray(geom.vertices, dtype=np.float64)

    colors = None
    if getattr(geom, "colors", None) is not None and len(geom.colors) == len(xyz):
        colors = np.asarray(geom.colors)
    elif getattr(getattr(geom, "visual", None), "vertex_colors", None) is not None:
        vc = np.asarray(geom.visual.vertex_colors)
        if len(vc) == len(xyz):
            colors = vc

    if colors is None:
        rgb = np.full((xyz.shape[0], 3), 128.0)  # neutral gray if no color present
    else:
        rgb = colors[:, :3].astype(np.float64)

    return np.concatenate([xyz, rgb], axis=1)
