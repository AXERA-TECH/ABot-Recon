"""Reference per-frame runners (encoder / decoder_step / heads as three separate sessions).

Both expose the same 3 methods so the assembly in backend.py is identical:
  run_encoder(image[1,3,280,504] f32)   -> patch_tokens[1,720,1024] f32
  run_decoder(feed: dict)               -> {fused_hidden, present_key, present_value, present_valid}
  run_heads(fused_hidden[1,725,2048])   -> {local_points, camera_features, confidence}

- PyAxEngineRunner : pyaxengine InferenceSession on the NPU (AXCL card or on-chip, whichever
                     provider pyaxengine finds). Every call copies the full KV cache host<->device,
                     so it is ~3x slower than native_runner; kept as the numeric reference
                     (scripts/validate_native.py) and as a fallback (ABOT_RUNNER=pyaxengine).
- OnnxRunner       : onnxruntime CPU on the delivery ONNX golden (scripts/validate_onnx.py).
"""
from __future__ import annotations

import os


class PyAxEngineRunner:
    provider = "pyaxengine"

    def __init__(self, model_dir: str, device_id: int = 0, suffix: str = "_kitti02"):
        import axengine
        from axengine import InferenceSession

        avail = list(getattr(axengine, "get_available_providers", lambda: [])())
        if "AXCLRTExecutionProvider" in avail:
            kw = dict(providers=["AXCLRTExecutionProvider"], provider_options=[{"device_id": device_id}])
        elif avail:
            kw = dict(providers=[avail[0]])
        else:
            kw = {}

        def mk(name: str):
            return InferenceSession(os.path.join(model_dir, f"{name}{suffix}.axmodel"), **kw)

        self._enc = mk("encoder")
        self._dec = mk("decoder_step")
        self._heads = mk("heads")
        self._dec_out = [a.name for a in self._dec.get_outputs()]
        self._heads_out = [a.name for a in self._heads.get_outputs()]
        self.device_id = device_id

    def run_encoder(self, image):
        return self._enc.run(None, {"image": image})[0]

    def run_decoder(self, feed):
        return dict(zip(self._dec_out, self._dec.run(None, feed)))

    def run_heads(self, fused_hidden):
        return dict(zip(self._heads_out, self._heads.run(None, {"fused_hidden": fused_hidden})))


class OnnxRunner:
    provider = "onnx"

    def __init__(self, onnx_dir: str, intra_op: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if intra_op:
            so.intra_op_num_threads = int(intra_op)

        def mk(name: str):
            return ort.InferenceSession(os.path.join(onnx_dir, f"{name}.onnx"), sess_options=so,
                                        providers=["CPUExecutionProvider"])

        self._enc = mk("encoder")
        self._dec = mk("decoder_step")
        self._heads = mk("heads")
        self._dec_out = [o.name for o in self._dec.get_outputs()]
        self._heads_out = [o.name for o in self._heads.get_outputs()]

    def run_encoder(self, image):
        return self._enc.run(None, {"image": image})[0]

    def run_decoder(self, feed):
        return dict(zip(self._dec_out, self._dec.run(None, feed)))

    def run_heads(self, fused_hidden):
        return dict(zip(self._heads_out, self._heads.run(None, {"fused_hidden": fused_hidden})))
