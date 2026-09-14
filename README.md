# ABot-Recon on Axera NPU

[ABot-Recon](https://huggingface.co/acvlab/ABot-Recon) 在 Axera NPU 上的推理实现:输入一段视频,输出相机位姿、世界坐标点云和置信度,并附带一个建图服务(网页上传视频 → 点云 / 户型俯视图 / 3D 查看)。

支持两种运行形态,同一套代码:

| 形态 | 运行时 | 设备 |
|---|---|---|
| AXCL | libaxcl_rt | 主机 + AX650N PCIe 卡 |
| AX650 片上 | libax_engine / libax_sys | AX650N 板 |

主机侧全部为 numpy 实现(预处理、位姿头、点云),不依赖 torch / open3d。

## 目录结构

```
abot_axera/                 推理核心
  native_runner.py          encoder → decoder_step → heads 链式推理,KV cache 常驻设备(AXCL / 片上)
  runners.py                参考实现:pyaxengine 逐模型 InferenceSession / ONNX Runtime
  backend.py                AbotRecon:图片序列 → 位姿 / 世界点 / 置信度
  pose_head.py              主机侧 AdjacentPoseHead(numpy)
  preprocess.py + csrc/     帧预处理(C 缩放内核,cffi 调用;prebuilt/ 内含 aarch64 预编译)
  pointcloud.py             体素下采样、PLY 读写
  progress.py               任务进度与预估剩余时间
service/                    FastAPI 服务 + viser 3D + 网页面板
scripts/                    验证与基准脚本
```

## 模型与依赖

| 内容 | 环境变量 | 说明 |
|---|---|---|
| `encoder_*.axmodel` `decoder_step_*.axmodel` `heads_*.axmodel` | `ABOT_MODELS`、`ABOT_MODEL_SUFFIX`(默认 `_kitti02`) | pulsar2 编译的 AX650N 模型,共约 1.4 GB |
| `host_pose_head/pose_head.safetensors`、`pose_head_config.json` | `ABOT_POSE_WEIGHTS`、`ABOT_POSE_CONFIG`(或 `ABOT_DELIVERY` 交付包目录) | 主机侧位姿头 |
| `models/onnx/*.onnx` | `ABOT_DELIVERY` | 运行不需要,仅 `scripts/validate_onnx.py` 评估量化误差时使用 |

Python 依赖见 `requirements.txt`:推理只需 numpy、cffi、Pillow;服务另需 opencv、matplotlib、viser、fastapi、uvicorn。Axera 运行时与 pyaxengine 来自 SDK。

缩放内核在首次导入时用 gcc 编译到 `abot_axera/_build/`;没有编译器时使用 `abot_axera/prebuilt/` 中的预编译库(aarch64),都没有时退回 numpy 实现。

## 运行

```bash
bash start_service.sh                                              # 前台
tmux new-session -d -s abot "bash start_service.sh > service_run.log 2>&1"   # 常驻
```

- 网页与 API:`http://<host>:8011`;viser 3D:`:8082`(内嵌在网页中)
- 网页流程:上传视频 → 选抽帧率 → 提交 → 进度条(帧 i/N、已用、预计剩余)→ 3D 查看 / 俯视图 / 下载 ply
- 模型在首次任务时加载,空闲 `MAP_IDLE_UNLOAD` 秒后卸载

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `ABOT_RUNNER` | `native` | `native`:KV 常驻设备;`pyaxengine`:逐模型 InferenceSession(参考实现) |
| `ABOT_DEVICE` | `auto` | `axcl` / `ax650` / `auto` |
| `ABOT_DEVICE_ID` | `0` | AXCL 卡号(片上忽略) |
| `ABOT_MODELS` / `ABOT_MODEL_SUFFIX` | `/home/axera/ABot-Recon` / `_kitti02` | axmodel 目录与文件名后缀 |
| `ABOT_DELIVERY` | 交付包路径 | `host_pose_head/` 所在目录 |
| `MAP_PORT` / `VISER_PORT` | `8011` / `8082` | 服务端口 |
| `MAP_DATA` | `./jobs` | 任务产物目录 |
| `MAP_FPS` | `8` | 默认抽帧率(网页可改,1–15) |
| `MAP_IDLE_UNLOAD` | `100000` | 空闲多少秒卸载模型 |

### API

```
POST /jobs                file[, fps, ceiling_cut, ceiling_keep]  → {job_id}
GET  /jobs                任务列表;运行中的带 progress {phase, frame, total, elapsed_s, sec_per_frame, eta_s}
GET  /jobs/{id}           任务状态、meta、产物
GET  /jobs/{id}/{file}    下载产物(floorplan.png / cloud.ply / recon.npz / view_*.png)
POST /jobs/{id}/view      载入 viser;POST /jobs/{id}/rerun 重跑;DELETE /jobs/{id} 删除
GET  /status              模型状态;POST /models/load|unload
```

## 资源消耗

模型输入 280×504,三个 axmodel 逐帧串行。

### CMM(AX650N,axcl-smi 实测)

| 项 | CMM |
|---|---|
| encoder | 337 MiB |
| decoder_step | 2513 MiB |
| heads | 217 MiB |
| KV cache 缓冲(4 × 588 MB fp32) | 2243 MiB |
| 合计(常驻) | 5341 MiB |

AXCL 卡 CMM 为 7040 MiB。片上运行需要同样的约 5.3 GB CMM;CMM 为 4–4.6 GB 的开发板放不下完整链路(只验证了 encoder + heads)。可选方案:调大板子的 CMM 预留至 6 GB 以上;将 decoder_step 的 KV 输入输出改为 fp16 重新导出(缓冲减半);KV 就地更新(需确认模型内部读写顺序)。

### 耗时(AX650N)

| 环节 | AXCL 卡 | AX650 板 |
|---|---|---|
| encoder | 0.19 s | 0.19 s |
| decoder_step | 2.63 s | 未测 |
| heads | 0.13 s | 0.13 s |
| 每帧合计(含预处理、位姿头) | 3.0 s | — |
| 模型加载 | 23 s | — |
| 预处理(JPEG 解码 + 缩放) | ~5 ms | ~45 ms |
| 位姿头 | ~10 ms | ~64 ms |
| 后处理(346 帧:位姿、点云、渲染) | ~40 s | — |

`ABOT_RUNNER=pyaxengine` 时每帧约 9.5 s。

估算:`总时长 ≈ 模型加载 + 抽帧 + N × 3.0 s + 后处理(≈ 0.13 s × N)`,N ≈ 视频秒数 × 抽帧率。例:346 帧(43 s 视频,8 fps)任务总时 1084 s。

## 验证脚本

```bash
python scripts/validate_native.py --runner pyaxengine --dev 0 --dump /tmp/ref.npz   # 参考实现
python scripts/validate_native.py --runner native     --dev 0 --dump /tmp/nat.npz   # 原生 runner
python scripts/validate_native.py --compare /tmp/ref.npz /tmp/nat.npz              # 逐位比对
python scripts/validate_onnx.py testdata/frames12 12                               # 对 ONNX golden(需 onnxruntime)
python scripts/onchip_check.py --make-ref ref.npz                                  # 卡上生成参考
python3 scripts/onchip_check.py /path/to/axmodels ref.npz                          # 板上比对 encoder + heads
bash scripts/http_smoke.sh video.mp4                                               # 服务端到端
```

## 已知限制

- 无回环闭合;长序列 revisit 的全局漂移不纠正,靠相机轨迹范围裁剪兜底。
- `present_valid` 封顶 8(应为 11),长序列有轻微漂移。
- 服务单任务串行。
