"""Host-side AdjacentPoseHead in plain numpy (fp32), streaming one frame at a time.

A line-by-line port of the released torch module (delivery host_pose_head/adjacent_pose_head.py):
frame descriptor MLP on the 5 pose tokens -> pair MLP -> delta translation + quaternion; a
temporal rotation refiner (single-query multi-head attention over the two frames' image tokens,
depthwise causal conv over the last 10 fused features) applies a bounded rotation residual.
Poses are composed sequentially: pose_t = pose_{t-1} @ delta_t, pose_0 = identity.

Weights come straight from the safetensors file (parsed here, no safetensors package needed).
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

_LN_EPS = 1e-5


def load_safetensors(path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader (F32/F16/BF16/I64 → numpy)."""
    dt = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "U8": np.uint8}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(blob[a:b], np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(blob[a:b], dt[meta["dtype"]])
        out[name] = arr.reshape(meta["shape"]).copy()
    return out


def _layernorm(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + _LN_EPS) * w + b


def _linear(x, w, b):
    return x @ w.T + b


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _mlp4(x, W, prefix):
    """nn.Sequential(LayerNorm, Linear, ReLU, Linear, ReLU) with keys prefix.{0,1,3}."""
    x = _layernorm(x, W[f"{prefix}.0.weight"], W[f"{prefix}.0.bias"])
    x = _relu(_linear(x, W[f"{prefix}.1.weight"], W[f"{prefix}.1.bias"]))
    return _relu(_linear(x, W[f"{prefix}.3.weight"], W[f"{prefix}.3.bias"]))


def quat_to_mat(q, eps=1e-8):
    """scalar-last quaternion [..., 4] -> rotation [..., 3, 3] (torch F.normalize semantics)."""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)
    i, j, k, r = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.maximum((q * q).sum(-1), eps)
    m = np.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], -1)
    return m.reshape(q.shape[:-1] + (3, 3))


def rotvec_to_mat(v, eps=1e-8):
    theta2 = (v * v).sum(-1, keepdims=True)
    theta2_safe = np.maximum(theta2, eps * eps)
    theta = np.sqrt(theta2_safe)
    theta4 = theta2 * theta2
    small = theta2 < eps * eps
    a = np.where(small, 1.0 - theta2 / 6.0 + theta4 / 120.0, np.sin(theta) / theta)
    b = np.where(small, 0.5 - theta2 / 24.0 + theta4 / 720.0, (1.0 - np.cos(theta)) / theta2_safe)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    skew = np.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(v.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3, dtype=v.dtype), skew.shape)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


class AdjacentPoseHead:
    """Streaming pose head. Call step(camera_features[725,512]) once per frame in order."""

    def __init__(self, weights: str | Path, config: str | Path):
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        assert cfg.get("rotation_format", "quat") == "quat" and cfg.get("translation_param", "vector") == "vector"
        self.num_pose_tokens = int(cfg.get("num_pose_tokens", 5))
        self.kernel = int(cfg.get("rot_correction_kernel", 10))
        self.max_rad = float(cfg.get("rot_correction_max_deg", 2.0)) * math.pi / 180.0
        self.use_age = bool(cfg.get("rot_correction_use_age_embed", True))
        self.num_heads = 8
        W = load_safetensors(weights)
        self.W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}
        self.hidden = self.W["rot_correction.frame_proj.weight"].shape[0]
        self.reset()

    # ---- state ----
    def reset(self):
        self.prev_desc = None
        self.prev_tokens = None
        self.prev_pose = np.eye(4, dtype=np.float32)
        self.buffer: list[np.ndarray] = []   # last <= kernel fused features (rotation refiner)
        self.n = 0

    # ---- pieces ----
    def _describe(self, feat):
        d = _mlp4(feat[: self.num_pose_tokens], self.W, "frame_descriptor")  # [5, hidden]
        return d.mean(0)

    def _predict_delta(self, prev, curr):
        W = self.W
        pair = np.concatenate([prev, curr, curr - prev, curr * prev])
        h = _relu(_linear(pair, W["pair_mlp.0.weight"], W["pair_mlp.0.bias"]))
        h = _relu(_linear(h, W["pair_mlp.2.weight"], W["pair_mlp.2.bias"]))
        t = _linear(h, W["delta_t_head.weight"], W["delta_t_head.bias"])
        q = _linear(h, W["delta_q_head.weight"], W["delta_q_head.bias"])
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = quat_to_mat(q)
        delta[:3, 3] = t
        return delta

    def _rotation_residual(self, prev_desc, curr_desc, prev_tok, curr_tok):
        W, p = self.W, "rot_correction"
        desc_feat = _mlp4(np.concatenate([prev_desc, curr_desc, curr_desc - prev_desc, curr_desc * prev_desc]), W, f"{p}.desc_proj")
        pm, cm = prev_tok.mean(0), curr_tok.mean(0)
        query = _mlp4(np.concatenate([pm, cm, cm - pm, cm * pm]), W, f"{p}.frame_query_proj")  # [hidden]
        memory = _linear(np.concatenate([prev_tok, curr_tok], 0), W[f"{p}.frame_proj.weight"], W[f"{p}.frame_proj.bias"])
        role = W[f"{p}.frame_role_embed"]
        memory[: len(prev_tok)] += role[0]
        memory[len(prev_tok):] += role[1]
        # nn.MultiheadAttention (batch_first, 1 query, no mask)
        H, D = self.num_heads, self.hidden // self.num_heads
        wq, wk, wv = np.split(W[f"{p}.frame_attn.in_proj_weight"], 3, 0)
        bq, bk, bv = np.split(W[f"{p}.frame_attn.in_proj_bias"], 3, 0)
        q = (_linear(query, wq, bq)).reshape(H, D)
        k = (_linear(memory, wk, bk)).reshape(-1, H, D)  # [L,H,D]
        v = (_linear(memory, wv, bv)).reshape(-1, H, D)
        att = _softmax(np.einsum("hd,lhd->hl", q, k) / math.sqrt(D), axis=-1)  # [H,L]
        ctx = np.einsum("hl,lhd->hd", att, v).reshape(-1)
        out = _linear(ctx, W[f"{p}.frame_attn.out_proj.weight"], W[f"{p}.frame_attn.out_proj.bias"])
        temporal = _layernorm(query + out, W[f"{p}.frame_norm.weight"], W[f"{p}.frame_norm.bias"])
        fused = _mlp4(np.concatenate([desc_feat, temporal]), W, f"{p}.fuse_proj")  # [hidden]

        self.buffer = (self.buffer + [fused])[-self.kernel:]
        K = self.kernel
        window = np.zeros((K, self.hidden), np.float32)
        pad = K - len(self.buffer)
        window[pad:] = np.stack(self.buffer)
        if self.use_age:
            age = W[f"{p}.age_embed.weight"][np.arange(K - 1, -1, -1)]  # position t gets id K-1-t
            window[pad:] += age[pad:]
        # depthwise Conv1d(kernel=K) over the window -> one output step per channel
        conv = (window.T * W[f"{p}.conv.weight"][:, 0, :]).sum(1) + W[f"{p}.conv.bias"]
        gate = (window.T * W[f"{p}.gate.weight"][:, 0, :]).sum(1) + W[f"{p}.gate.bias"]
        hidden = conv * _sigmoid(gate)
        return self.max_rad * np.tanh(_linear(hidden, W[f"{p}.out.weight"], W[f"{p}.out.bias"]))

    # ---- public ----
    def step(self, camera_features: np.ndarray) -> np.ndarray:
        """camera_features [tokens, C] (or [1, tokens, C]) of the next frame -> its c2w pose [4,4]."""
        feat = np.asarray(camera_features, dtype=np.float32)
        if feat.ndim == 3:
            feat = feat[0]
        desc = self._describe(feat)
        tokens = feat[self.num_pose_tokens:]
        if self.prev_desc is None:            # frame 0: identity
            pose = np.eye(4, dtype=np.float32)
        else:
            delta = self._predict_delta(self.prev_desc, desc)
            resid = self._rotation_residual(self.prev_desc, desc, self.prev_tokens, tokens)
            delta[:3, :3] = delta[:3, :3] @ rotvec_to_mat(resid)
            pose = (self.prev_pose @ delta).astype(np.float32)
        self.prev_desc, self.prev_tokens, self.prev_pose = desc, tokens, pose
        self.n += 1
        return pose

    def run(self, camera_features_seq) -> np.ndarray:
        """[N, tokens, C] -> [N, 4, 4] (resets first)."""
        self.reset()
        return np.stack([self.step(f) for f in camera_features_seq])
