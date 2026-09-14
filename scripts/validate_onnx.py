"""Validate the NPU chain numerically against the ONNX golden chain.

Same assembly code (abot_axera.NpuReleasedModel) run with two runners:
  - make_runner() : kitti02 axmodels on the NPU (ABOT_RUNNER native|pyaxengine, ABOT_DEVICE auto|axcl|ax650)
  - OnnxRunner    : delivery models/onnx/*.onnx on CPU
Both feed the SAME host AdjacentPoseHead, so any gap isolates the runtime.
Reports cos / relL2 on local_points, camera_features-derived poses, world_points, confidence.

Usage: scripts/validate_onnx.py <frames_dir_with_jpgs> [num_frames]
"""
import os, sys, glob, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from abot_axera.backend import NpuReleasedModel, load_pose_head, make_runner
from abot_axera.runners import OnnxRunner

DELIV = os.environ.get("ABOT_DELIVERY", "/home/axera/abot650/ABot-Recon_AX650_交付包_20260907")
POSE_W = os.path.join(DELIV, "host_pose_head", "pose_head.safetensors")
POSE_C = os.path.join(DELIV, "host_pose_head", "pose_head_config.json")
ONNX_DIR = os.path.join(DELIV, "models", "onnx")
MODELS = os.environ.get("ABOT_MODELS", "/home/axera/ABot-Recon")
DEV = int(os.environ.get("ABOT_DEVICE_ID", "0"))


def _flat(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def cos(a, b):
    a, b = _flat(a, b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / (d + 1e-12))


def relL2(a, b):
    a, b = _flat(a, b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    fdir = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("NF", "12"))
    frames = sorted(glob.glob(os.path.join(fdir, "*.jpg")))[:nf]
    if not frames:
        frames = sorted(glob.glob(os.path.join(fdir, "*.png")))[:nf]
    print(f"frames: {len(frames)} from {fdir}")
    assert frames, "no frames"

    pose = load_pose_head(POSE_W, POSE_C)  # shared -> isolates runner difference

    kw = dict(output_local_points=True, output_world_points=True, output_confidence=True)

    print("== NPU chain ==", flush=True)
    ax = NpuReleasedModel(make_runner(MODELS, device_id=DEV), pose)
    t = time.time()
    rax = ax.infer_paths(frames, **kw)
    tax = time.time() - t
    print(f"npu infer {tax:.1f}s  ({tax/len(frames):.2f}s/frame)", flush=True)

    print("== ONNX chain ==", flush=True)
    on = NpuReleasedModel(OnnxRunner(ONNX_DIR), pose)
    t = time.time()
    ron = on.infer_paths(frames, **kw)
    ton = time.time() - t
    print(f"onnx infer {ton:.1f}s", flush=True)

    print("\n== numeric parity (NPU vs ONNX) ==")
    for k in ["local_points", "camera_poses", "world_points", "confidence"]:
        a = rax[k].numpy()
        b = ron[k].numpy()
        print(f"{k:14s} shape={tuple(a.shape)}  cos={cos(a, b):.6f}  relL2={relL2(a, b):.5f}")

    # camera-center trajectory drift (meters, model units)
    ca = rax["camera_poses"].numpy()[:, :3, 3]
    cb = ron["camera_poses"].numpy()[:, :3, 3]
    d = np.linalg.norm(ca - cb, axis=1)
    print(f"\ncam-center L2 per frame: mean={d.mean():.5f} max={d.max():.5f}")
    print("pose[0] (should be identity):\n", np.round(rax["camera_poses"].numpy()[0], 4))
    print("pose[-1] npu trans:", np.round(ca[-1], 4), " onnx trans:", np.round(cb[-1], 4))


if __name__ == "__main__":
    main()
