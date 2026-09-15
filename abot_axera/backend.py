"""ABot-Recon inference on an Axera NPU, torch-free.

Streaming chain (delivery docs/模型说明_中文.md):
  frame -> preprocess[1,3,280,504] -> encoder -> patch_tokens[1,720,1024]
        -> decoder_step (+right-aligned KV-cache, present_* fed straight back)
        -> fused_hidden[1,725,2048]
        -> heads -> local_points[1,280,504,3], camera_features[1,725,512], confidence logits
  then the host AdjacentPoseHead (numpy) streams camera_features -> c2w camera_poses[N,4,4]
  world_points = R @ local_points + t ; confidence = sigmoid(logits)

Runner flavours (ABOT_RUNNER):
  native      (default) native_runner.NativeChainRunner — KV cache and the intermediates stay on
              the device; ~3 s/frame on AX650N. Same code on a PCIe AXCL card (libaxcl_rt) and
              on-chip AX650 (libax_engine); ABOT_DEVICE=auto|axcl|ax650 picks the runtime.
  pyaxengine  runners.PyAxEngineRunner — one pyaxengine InferenceSession per model; every call
              round-trips the 2x588 MB KV through the host (~9.5 s/frame). Reference / fallback.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import progress
from .pose_head import AdjacentPoseHead

KV_SHAPE = (1, 18, 16, 11, 725, 64)


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


def transform_local_points(local_points: np.ndarray, camera_poses: np.ndarray) -> np.ndarray:
    """[N,H,W,3] camera-frame points + [N,4,4] c2w -> world points."""
    R = camera_poses[:, :3, :3]
    t = camera_poses[:, :3, 3]
    return np.einsum("nij,nhwj->nhwi", R, local_points) + t[:, None, None]


def relative_from_c2w(camera_poses: np.ndarray) -> np.ndarray:
    """Adjacent transforms mapping frame t coordinates into frame t+1: inv(P[t+1]) @ P[t]."""
    if len(camera_poses) < 2:
        return np.empty((0, 4, 4), camera_poses.dtype)
    return np.linalg.solve(camera_poses[1:], camera_poses[:-1])


@dataclass
class ReconResult:
    camera_poses: np.ndarray                 # [N,4,4] c2w, frame 0 = identity
    relative_poses: np.ndarray               # [N-1,4,4]
    local_points: np.ndarray | None = None   # [M,H,W,3]
    world_points: np.ndarray | None = None   # [M,H,W,3]
    confidence: np.ndarray | None = None     # [M,H,W] in (0,1)
    colors: np.ndarray | None = None         # [M,H,W,3] uint8 (output_colors=True)
    metadata: dict = field(default_factory=dict)


class AbotRecon:
    """Streaming reconstruction: image paths -> poses / world points / confidence (numpy)."""

    def __init__(self, runner, pose_head: AdjacentPoseHead, *, height: int = 280, width: int = 504):
        self.runner = runner
        self.pose = pose_head
        self.height, self.width = int(height), int(width)
        self.provider = getattr(runner, "provider", "npu")
        self._native = hasattr(runner, "step") and hasattr(runner, "reset")

    # ---- per-frame chain: native runner or three sessions, same outputs ----
    def _frame_native(self, image, fi, _state):
        return self.runner.step(image, fi), None

    def _frame_sessions(self, image, fi, state):
        past_key, past_value, past_valid = state
        patch_tokens = self.runner.run_encoder(image)
        dec = self.runner.run_decoder({
            "patch_tokens": np.ascontiguousarray(patch_tokens, dtype=np.float32),
            "past_key": past_key, "past_value": past_value, "past_valid": past_valid,
            "frame_index": np.array([fi], np.float32),
        })
        state = tuple(np.ascontiguousarray(dec[k], dtype=np.float32)
                      for k in ("present_key", "present_value", "present_valid"))
        return self.runner.run_heads(np.ascontiguousarray(dec["fused_hidden"], dtype=np.float32)), state

    def infer(self, frames, *, output_local_points: bool = False, output_world_points: bool = True,
              output_confidence: bool = True, output_colors: bool = False, dense_output_indices=None,
              total: int | None = None, loop_closure: bool = False, **_ignored) -> ReconResult:
        """frames: image paths, or an iterable of (chw float32 [3,280,504], rgb uint8 [280,504,3]) pairs
        (see abot_axera.video). `total` gives the frame count for progress when frames is a generator."""
        if loop_closure:
            print("[abot] loop_closure requested but not available on the NPU path; ignored", flush=True)
        want_points = bool(output_local_points or output_world_points)
        if hasattr(frames, "__len__"):
            total = len(frames)
        keep_set = None if dense_output_indices is None else {int(i) for i in dense_output_indices}

        if self._native:
            self.runner.reset()
            step, state = self._frame_native, None
        else:
            step = self._frame_sessions
            state = (np.zeros(KV_SHAPE, np.float32), np.zeros(KV_SHAPE, np.float32), np.zeros((1,), np.float32))

        self.pose.reset()
        poses, local, conf, colors = [], {}, {}, {}
        n = int(total) if total else -1
        progress.start_infer(max(n, 0), hint_spf=3.0 if self._native else 9.5)
        print(f"[abot] infer start: {n} frames via {self.provider}", flush=True)
        for fi, item in enumerate(self._iter_frames(frames)):
            chw, rgb = item
            t0 = time.time()
            heads, state = step(chw[None], fi, state)
            poses.append(self.pose.step(heads["camera_features"][0]))
            if keep_set is None or fi in keep_set:
                if want_points:
                    local[fi] = np.ascontiguousarray(heads["local_points"][0], dtype=np.float32)
                if output_confidence:
                    conf[fi] = np.ascontiguousarray(heads["confidence"][0], dtype=np.float32)
                if output_colors:
                    colors[fi] = rgb
            progress.upd(fi + 1, n if n > 0 else None)
            print(f"[abot] frame {fi + 1}/{n} done in {time.time() - t0:.1f}s", flush=True)
        progress.set_phase("post")
        n = len(poses)
        if n == 0:
            raise ValueError("no frames")
        keep = list(range(n)) if keep_set is None else sorted(i for i in keep_set if i < n)

        camera_poses = np.stack(poses).astype(np.float32)
        res = ReconResult(camera_poses=camera_poses, relative_poses=relative_from_c2w(camera_poses),
                          metadata={"frames": n, "provider": self.provider, "loop_closure": False,
                                    "dense_output_indices": keep})
        if want_points:
            lp = np.stack([local[i] for i in keep])
            if output_local_points:
                res.local_points = lp
            if output_world_points:
                res.world_points = transform_local_points(lp, camera_poses[keep])
        if output_confidence:
            logits = np.stack([conf[i] for i in keep])
            if logits.ndim == 4 and logits.shape[-1] == 1:
                logits = logits[..., 0]
            res.confidence = 1.0 / (1.0 + np.exp(-logits))
        if output_colors:
            res.colors = np.stack([colors[i] for i in keep])
        return res

    def _iter_frames(self, frames):
        """Normalise the input: paths -> preprocessed pairs; pairs pass through."""
        from .preprocess import preprocess_image
        for item in frames:
            if isinstance(item, (str, Path)):
                chw, _ = preprocess_image(__import__("PIL.Image", fromlist=["Image"]).open(item))
                rgb = np.round(np.clip(chw, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
                yield chw, np.ascontiguousarray(rgb)
            else:
                yield item


def build_abot_recon(*, model_dir: str, pose_weights: str, pose_config: str, device_id: int = 0,
                     suffix: str = "_kitti02", runner=None, runner_kind: str | None = None,
                     device: str | None = None) -> AbotRecon:
    if runner is None:
        runner = make_runner(model_dir, device_id, suffix, runner_kind, device)
    return AbotRecon(runner, AdjacentPoseHead(pose_weights, pose_config))
