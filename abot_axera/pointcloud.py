"""Tiny point-cloud helpers (numpy only): voxel down-sampling and binary PLY read/write.

voxel_down_sample follows Open3D's semantics (voxel grid anchored at min_bound - voxel/2,
per-voxel mean of points and colors) so cloud sizes match what the service produced before.
"""
from __future__ import annotations

import numpy as np


class VoxelGrid:
    """Voxel assignment of a point set (Open3D voxel_down_sample semantics: grid anchored at
    min_bound - voxel/2, per-voxel mean). Computed once, then `mean()` averages any per-point
    attribute (points, several color sets) without redoing the sort."""

    def __init__(self, points: np.ndarray, voxel: float):
        P = np.asarray(points, np.float64)
        self.n = len(P)
        if self.n == 0:
            self.inv = np.zeros(0, np.int64); self.counts = np.zeros(0, np.int64); return
        origin = P.min(0) - voxel * 0.5
        idx = np.floor((P - origin) / voxel).astype(np.int64)
        dims = idx.max(0) + 1
        key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]      # unique 1-D key per voxel
        _, self.inv, self.counts = np.unique(key, return_inverse=True, return_counts=True)
        self.inv = self.inv.reshape(-1)

    def __len__(self):
        return len(self.counts)

    def mean(self, attr: np.ndarray) -> np.ndarray:
        A = np.asarray(attr, np.float64).reshape(self.n, -1)
        out = np.empty((len(self.counts), A.shape[1]), np.float64)
        for c in range(A.shape[1]):
            out[:, c] = np.bincount(self.inv, weights=A[:, c], minlength=len(self.counts))
        return (out / self.counts[:, None]).astype(np.float32)


def voxel_down_sample(points: np.ndarray, colors: np.ndarray, voxel: float):
    if len(points) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    g = VoxelGrid(points, voxel)
    return g.mean(points), g.mean(colors)


def write_ply(path, points: np.ndarray, colors01: np.ndarray) -> None:
    """Binary little-endian PLY: float x y z + uchar r g b. colors in [0,1]."""
    P = np.asarray(points, np.float32)
    C = (np.clip(np.asarray(colors01), 0, 1) * 255 + 0.5).astype(np.uint8)
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
    rec["r"], rec["g"], rec["b"] = C[:, 0], C[:, 1], C[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\ncomment ABot-Recon on Axera NPU\n"
              f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii")); f.write(rec.tobytes())


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "ushort": "<u2", "short": "<i2", "uint": "<u4", "int": "<i4"}


def read_ply(path):
    """Read a binary little-endian PLY vertex element -> (points float32 [N,3], colors uint8 [N,3] or None).
    Handles both this writer's float layout and Open3D's double layout."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError("bad PLY header")
            header += line
        lines = header.decode("ascii", "ignore").splitlines()
        if "format binary_little_endian" not in header.decode("ascii", "ignore"):
            raise ValueError("only binary_little_endian PLY supported")
        n, fields, in_vertex = 0, [], False
        for ln in lines:
            t = ln.split()
            if not t:
                continue
            if t[0] == "element":
                in_vertex = t[1] == "vertex"
                if in_vertex:
                    n = int(t[2])
            elif t[0] == "property" and in_vertex:
                if t[1] == "list":
                    raise ValueError("list properties in vertex element not supported")
                fields.append((t[2], _PLY_TYPES[t[1]]))
        rec = np.frombuffer(f.read(int(np.dtype(fields).itemsize) * n), dtype=np.dtype(fields), count=n)
    pts = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    names = rec.dtype.names
    col = None
    if all(c in names for c in ("red", "green", "blue")):
        col = np.stack([rec["red"], rec["green"], rec["blue"]], 1)
        col = (np.clip(col, 0, 1) * 255).astype(np.uint8) if col.dtype.kind == "f" else col.astype(np.uint8)
    return pts, col
