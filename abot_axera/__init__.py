"""ABot-Recon on Axera NPUs (AXCL PCIe cards and on-chip AX650) — numpy + cffi only, no torch.

Modules
  native_runner  device-resident chain runner (encoder -> decoder_step -> heads), KV cache
                 never leaves the device. libaxcl_rt (card) or libax_engine (on-chip).
  runners        reference runners: pyaxengine InferenceSession per model, ONNX Runtime golden.
  pose_head      host-side AdjacentPoseHead (numpy port of the released torch module).
  preprocess     width-lock antialiased bicubic resize + crop/pad (matches torchvision).
  backend        AbotRecon: image paths -> camera poses / world points / confidence.
  pointcloud     voxel down-sampling + binary PLY read/write.
  progress       per-frame progress + ETA for the web dashboard.
"""
from __future__ import annotations

from .native_runner import NativeChainRunner, detect_device

__all__ = ["NativeChainRunner", "detect_device", "AbotRecon", "ReconResult", "build_abot_recon", "make_runner",
           "AdjacentPoseHead", "preprocess_image", "iter_preprocessed"]

_LAZY = {"AbotRecon": "backend", "ReconResult": "backend", "build_abot_recon": "backend", "make_runner": "backend",
         "AdjacentPoseHead": "pose_head", "preprocess_image": "preprocess", "iter_preprocessed": "preprocess"}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(name)
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)
