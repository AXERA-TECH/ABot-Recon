"""ABot-Recon on Axera NPUs (AXCL PCIe cards and on-chip AX650).

Modules
  native_runner  device-resident chain runner (encoder -> decoder_step -> heads), KV cache
                 never leaves the device. Works through libaxcl_rt (card) or libax_engine
                 (on-chip). No torch dependency — importable on a bare AX650 board.
  runners        reference runners: pyaxengine InferenceSession per model, ONNX Runtime golden.
  backend        NpuReleasedModel — drop-in replacement for the torch ReleasedABotReconModel
                 (same infer_paths contract) so upstream abot_recon.ABotRecon runs unchanged.
  pose_head      host-side AdjacentPoseHead (torch, CPU).
  progress       per-frame progress + ETA for the web dashboard.

`import abot_axera` stays light; the torch-dependent pieces load on first attribute access.
"""
from __future__ import annotations

from .native_runner import NativeChainRunner, detect_device

__all__ = ["NativeChainRunner", "detect_device", "NpuReleasedModel", "build_abot_recon",
           "make_runner", "load_pose_head"]

_LAZY = {"NpuReleasedModel": "backend", "build_abot_recon": "backend", "make_runner": "backend",
         "load_pose_head": "backend"}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(name)
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)
