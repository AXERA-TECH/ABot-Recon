"""Gaussian splats straight from the dense point maps (no training).

Every reconstructed point becomes a flat Gaussian lying on the surface: the normal comes from
the neighbouring points of the same frame's point map, the tangential extent from the spacing
to those neighbours, colour from the frame. Rendered as splats the surfaces close up and the
scene reads as a walk-through instead of a cloud of dots.

Outputs the standard 3DGS PLY layout (x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3),
readable by SuperSplat / PlayCanvas / most web splat viewers, plus arrays for viser's
add_gaussian_splats.
"""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814


def frame_geometry(wp: np.ndarray, cams: np.ndarray, st: int = 2):
    """Per-pixel normals and tangential spacing from world point maps.

    wp   : [M,H,W,3] world points (one map per frame)
    cams : [M,3] camera centres (to orient normals towards the camera)
    st   : sampling stride used for the cloud (points taken every `st` pixels)
    Returns (normals [M,H',W',3], spacing [M,H',W',2]) sampled on the same ::st grid as the cloud.
    """
    M, H, W, _ = wp.shape
    P = wp[:, ::st, ::st, :]
    # neighbours on the sampled grid (edge-replicated)
    right = np.concatenate([P[:, :, 1:], P[:, :, -1:]], axis=2)
    left = np.concatenate([P[:, :, :1], P[:, :, :-1]], axis=2)
    down = np.concatenate([P[:, 1:], P[:, -1:]], axis=1)
    up = np.concatenate([P[:, :1], P[:, :-1]], axis=1)
    du = right - left  # ~2 samples apart
    dv = down - up
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    to_cam = cams[:, None, None, :] - P
    flip = (np.sum(n * to_cam, axis=-1, keepdims=True) < 0)
    n = np.where(flip, -n, n)
    spacing = np.stack([np.linalg.norm(du, axis=-1) * 0.5, np.linalg.norm(dv, axis=-1) * 0.5], axis=-1)
    return n.astype(np.float32), spacing.astype(np.float32)


def build_gaussians(points, colors01, normals, spacing, opacity, *, voxel: float, max_scale_vox: float = 3.0):
    """Assemble per-point Gaussian parameters.

    points [N,3], colors01 [N,3] in [0,1], normals [N,3] unit, spacing [N,2] tangential half-spacing,
    opacity [N] in (0,1]. Splats larger than max_scale_vox voxels (depth discontinuities) are
    clamped. Returns dict with centers, rgb (uint8), opacity, scales [N,3], quat_wxyz [N,4], cov [N,3,3].
    """
    P = np.asarray(points, np.float32)
    n = np.asarray(normals, np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    # tangent frame: t1 ⟂ n from an arbitrary helper axis, t2 = n × t1
    helper = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1]], np.float32), np.array([[1, 0, 0]], np.float32))
    t1 = np.cross(helper, n); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True) + 1e-12
    t2 = np.cross(n, t1)
    s = np.asarray(spacing, np.float32)
    s_t = np.clip(np.maximum(s.mean(-1), 0.5 * voxel) * 0.9, 0.35 * voxel, max_scale_vox * voxel)  # isotropic in-plane sigma
    scales = np.stack([s_t, s_t, s_t * 0.15], -1).astype(np.float32)           # thin along the normal
    R = np.stack([t1, t2, n], -1)                                              # columns = local axes
    cov = np.einsum("nij,nj,nkj->nik", R, scales ** 2, R).astype(np.float32)
    quat = _mat_to_quat_wxyz(R)
    rgb = (np.clip(np.asarray(colors01), 0, 1) * 255 + 0.5).astype(np.uint8)
    # surfaces should be solid: opacity is a constant, the model's confidence only gates which
    # points exist (mapping_pipeline drops the low-confidence 45%)
    op = np.full(len(P), float(opacity) if np.isscalar(opacity) else 0.95, np.float32)
    return {"centers": P, "rgb": rgb, "opacity": op, "scales": scales, "quat": quat, "cov": cov}


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((len(R), 4), np.float32)
    a = tr > 0
    s = np.sqrt(np.maximum(tr[a] + 1.0, 1e-12)) * 2
    q[a, 0] = 0.25 * s; q[a, 1] = (m21[a] - m12[a]) / s; q[a, 2] = (m02[a] - m20[a]) / s; q[a, 3] = (m10[a] - m01[a]) / s
    b = (~a) & (m00 >= m11) & (m00 >= m22)
    s = np.sqrt(np.maximum(1.0 + m00[b] - m11[b] - m22[b], 1e-12)) * 2
    q[b, 0] = (m21[b] - m12[b]) / s; q[b, 1] = 0.25 * s; q[b, 2] = (m01[b] + m10[b]) / s; q[b, 3] = (m02[b] + m20[b]) / s
    c = (~a) & (~b) & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11[c] - m00[c] - m22[c], 1e-12)) * 2
    q[c, 0] = (m02[c] - m20[c]) / s; q[c, 1] = (m01[c] + m10[c]) / s; q[c, 2] = 0.25 * s; q[c, 3] = (m12[c] + m21[c]) / s
    d = (~a) & (~b) & (~c)
    s = np.sqrt(np.maximum(1.0 + m22[d] - m00[d] - m11[d], 1e-12)) * 2
    q[d, 0] = (m10[d] - m01[d]) / s; q[d, 1] = (m02[d] + m20[d]) / s; q[d, 2] = (m12[d] + m21[d]) / s; q[d, 3] = 0.25 * s
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def write_splat_ply(path, g: dict) -> None:
    """3DGS-format PLY (SH degree 0). Colours as f_dc, opacity/scales in the usual logit/log domains."""
    N = len(g["centers"])
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
              ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"), ("opacity", "<f4"),
              ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
              ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
    rec = np.zeros(N, dtype=fields)
    rec["x"], rec["y"], rec["z"] = g["centers"].T
    rec["nx"] = rec["ny"] = rec["nz"] = 0.0
    rgb = g["rgb"].astype(np.float32) / 255.0
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = ((rgb - 0.5) / SH_C0).T
    op = np.clip(g["opacity"], 1e-4, 1 - 1e-4)
    rec["opacity"] = np.log(op / (1 - op))
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = np.log(np.maximum(g["scales"], 1e-7)).T
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = g["quat"].T
    header = "ply\nformat binary_little_endian 1.0\ncomment ABot-Recon on Axera NPU (surfel splats)\n" \
             f"element vertex {N}\n" + "".join(f"property float {n}\n" for n, _ in fields) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii")); f.write(rec.tobytes())


def read_splat_ply(path):
    """-> (centers [N,3], rgb uint8 [N,3], opacity [N,1], cov [N,3,3]) for viser."""
    from .pointcloud import _PLY_TYPES
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        n, fields = 0, []
        for ln in header.decode("ascii", "ignore").splitlines():
            t = ln.split()
            if t and t[0] == "element" and t[1] == "vertex": n = int(t[2])
            elif t and t[0] == "property": fields.append((t[2], _PLY_TYPES[t[1]]))
        rec = np.frombuffer(f.read(np.dtype(fields).itemsize * n), dtype=np.dtype(fields), count=n)
    centers = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    rgb = np.clip(np.stack([rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"]], 1) * SH_C0 + 0.5, 0, 1)
    rgb = (rgb * 255 + 0.5).astype(np.uint8)
    op = 1 / (1 + np.exp(-rec["opacity"].astype(np.float32)))[:, None]
    scales = np.exp(np.stack([rec["scale_0"], rec["scale_1"], rec["scale_2"]], 1).astype(np.float32))
    q = np.stack([rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"]], 1).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    cov = np.einsum("nij,nj,nkj->nik", R, scales ** 2, R).astype(np.float32)
    return centers, rgb, op.astype(np.float32), cov
