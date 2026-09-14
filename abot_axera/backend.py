"""NPU-backed ABot-Recon model with the torch `infer_paths` output contract.

Streaming chain (delivery docs/模型说明_中文.md):
  frame -> preprocess[1,3,280,504] -> encoder -> patch_tokens[1,720,1024]
        -> decoder_step (+right-aligned KV-cache, present_* fed straight back)
        -> fused_hidden[1,725,2048]
        -> heads -> local_points[1,280,504,3], camera_features[1,725,512], confidence logits
  then host AdjacentPoseHead streams camera_features -> c2w camera_poses[N,4,4]
  world_points = R @ local_points + t   (abot_recon.geometry.transform_local_points)

Runner flavours (ABOT_RUNNER):
  native      (default) native_runner.NativeChainRunner — KV cache and the intermediates stay on
              the device; ~3 s/frame on AX650N. Same code on a PCIe AXCL card (libaxcl_rt) and
              on-chip AX650 (libax_engine); ABOT_DEVICE=auto|axcl|ax650 picks the runtime.
  pyaxengine  runners.PyAxEngineRunner — one pyaxengine InferenceSession per model; every call
              round-trips the 2x588 MB KV through the host (~9.5 s/frame). Reference / fallback.

The returned dict matches abot_recon.model.ReleasedABotReconModel.infer_paths, so upstream
ABotRecon.infer (api.py) — relative poses, confidence masking — runs verbatim on top.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import progress

KV_SHAPE = (1, 18, 16, 11, 725, 64)


def load_pose_head(weights, config, device: str = "cpu"):
    from safetensors.torch import load_file

    from .pose_head import AdjacentPoseHead

    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    cfg.pop("class", None)
    model = AdjacentPoseHead(**cfg).to(device).eval()
    model.load_state_dict(load_file(str(weights), device=device), strict=True)
    return model


def make_runner(model_dir: str, device_id: int = 0, suffix: str = "_kitti02",
                kind: str | None = None, device: str | None = None):
    """kind: native (default) | pyaxengine.  device (native only): auto | axcl | ax650."""
    kind = (kind or os.environ.get("ABOT_RUNNER", "native")).lower()
    device = (device or os.environ.get("ABOT_DEVICE", "auto")).lower()
    if kind == "native":
        from .native_runner import NativeChainRunner

        return NativeChainRunner(model_dir, device_id=device_id, suffix=suffix, device=device)
    if kind in ("pyaxengine", "axengine"):
        from .runners import PyAxEngineRunner

        return PyAxEngineRunner(model_dir, device_id=device_id, suffix=suffix)
    raise ValueError(f"ABOT_RUNNER={kind!r} (native | pyaxengine)")


class NpuReleasedModel:
    """Same interface as torch ReleasedABotReconModel; an NPU (or ONNX) runner inside."""

    def __init__(self, runner, pose_head, *, height: int = 280, width: int = 504):
        self.runner = runner
        self.pose = pose_head
        self.height = int(height)
        self.width = int(width)
        # attributes some upstream code probes for
        self.device_name = "cpu"
        self.attention_backend = getattr(runner, "provider", "npu")
        # native runner: one call per frame, KV never leaves the device
        self._native = hasattr(runner, "step") and hasattr(runner, "reset")

    def reset(self) -> None:
        # KV-cache and pose state are local to each infer_paths call; nothing to clear.
        return None

    # ---- per-frame chain: two flavours, same outputs ----
    def _frame_native(self, image: np.ndarray, fi: int, _state):
        return self.runner.step(image, fi), None

    def _frame_sessions(self, image: np.ndarray, fi: int, state):
        past_key, past_value, past_valid = state
        patch_tokens = self.runner.run_encoder(image)
        dec = self.runner.run_decoder({
            "patch_tokens": np.ascontiguousarray(patch_tokens, dtype=np.float32),
            "past_key": past_key, "past_value": past_value, "past_valid": past_valid,
            "frame_index": np.array([fi], np.float32),
        })
        # present_* fed straight back as next frame's past_* (right-aligned cache)
        state = tuple(np.ascontiguousarray(dec[k], dtype=np.float32)
                      for k in ("present_key", "present_value", "present_valid"))
        return self.runner.run_heads(np.ascontiguousarray(dec["fused_hidden"], dtype=np.float32)), state

    @torch.inference_mode()
    def infer_paths(self, paths, *, output_local_points: bool, output_world_points: bool,
                    output_confidence: bool, dense_output_indices=None, image_observer=None) -> dict:
        from abot_recon.geometry import transform_local_points
        from abot_recon.preprocessing import iter_preprocessed

        want_points = bool(output_local_points or output_world_points)

        if self._native:
            self.runner.reset()
            step, state = self._frame_native, None
        else:
            step = self._frame_sessions
            state = (np.zeros(KV_SHAPE, np.float32), np.zeros(KV_SHAPE, np.float32), np.zeros((1,), np.float32))

        cam_feats: list[np.ndarray] = []
        local_list: list[np.ndarray] = []
        conf_list: list[np.ndarray] = []

        n_total = len(paths) if hasattr(paths, "__len__") else -1
        progress.start_infer(n_total, hint_spf=3.0 if self._native else 9.5)
        print(f"[abot] infer start: {n_total} frames via {self.attention_backend}", flush=True)

        for fi, (tns, _) in enumerate(iter_preprocessed(paths, height=self.height, width=self.width)):
            t0 = time.time()
            if image_observer is not None:
                image_observer(tns.unsqueeze(0).unsqueeze(0))
            image = np.ascontiguousarray(tns.unsqueeze(0).numpy(), dtype=np.float32)
            heads, state = step(image, fi, state)
            cam_feats.append(np.ascontiguousarray(heads["camera_features"], dtype=np.float32))
            if want_points:
                local_list.append(np.ascontiguousarray(heads["local_points"][0], dtype=np.float32))
            if output_confidence:
                conf_list.append(np.ascontiguousarray(heads["confidence"][0], dtype=np.float32))
            progress.upd(fi + 1, n_total)
            print(f"[abot] frame {fi + 1}/{n_total} done in {time.time() - t0:.1f}s", flush=True)

        progress.set_phase("post")
        # host pose head: stream camera_features -> c2w poses (frame 0 = identity).
        # Poses are always composed over the FULL sequence (never subsampled).
        pstate, poses = None, []
        for feat_np in cam_feats:
            feat = torch.from_numpy(feat_np).float().unsqueeze(1)  # [1,1,725,512]
            pose, pstate = self.pose(feat, camera_state=pstate, return_state=True)
            poses.append(pose[:, 0])  # [1,4,4]
        camera_poses = torch.cat(poses, dim=0).float()  # [N,4,4]

        # dense outputs: torch returns local/world/conf only for dense_output_indices
        # (default = every frame). Mirror that so upstream api.py stays exact.
        n = len(cam_feats)
        keep = list(range(n)) if dense_output_indices is None else list(dense_output_indices)

        out: dict = {"camera_poses": camera_poses, "attention_backend": self.attention_backend}
        if want_points:
            local = torch.from_numpy(np.stack([local_list[i] for i in keep], axis=0)).float()  # [M,H,W,3]
            out["local_points"] = local
            if output_world_points:
                dense_poses = camera_poses[keep] if dense_output_indices is not None else camera_poses
                out["world_points"] = transform_local_points(local, dense_poses)
        if output_confidence:
            logits = torch.from_numpy(np.stack([conf_list[i] for i in keep], axis=0)).float()  # [M,H,W,1]
            if logits.ndim == 4 and logits.shape[-1] == 1:
                logits = logits[..., 0]
            out["confidence"] = torch.sigmoid(logits)  # [M,H,W]
        return out


def build_abot_recon(*, model_dir: str, pose_weights: str, pose_config: str, device_id: int = 0,
                     suffix: str = "_kitti02", runner=None, runner_kind: str | None = None,
                     device: str | None = None):
    """abot_recon.ABotRecon whose backbone runs on an Axera NPU (or the supplied runner)."""
    from abot_recon import ABotRecon, InferenceConfig

    if runner is None:
        runner = make_runner(model_dir, device_id, suffix, runner_kind, device)
    pose = load_pose_head(pose_weights, pose_config, device="cpu")
    model = NpuReleasedModel(runner, pose)
    config = InferenceConfig().override(
        device="cpu", amp_dtype="fp32", attention_backend="sdpa",
        output_local_points=True, output_world_points=True, output_confidence=True,
        loop_closure=False,
    )
    return ABotRecon(model, config)
