# ABot-Recon on Axera NPU

[ABot-Recon](https://huggingface.co/acvlab/ABot-Recon) 在 Axera NPU 上的推理实现:输入视频,输出相机位姿、世界坐标点云和置信度;附带建图服务(网页上传视频 → 点云 / 户型俯视图 / 3D 查看)。

支持两种运行形态:

| 形态 | 运行时 | 设备 |
|---|---|---|
| AXCL | libaxcl_rt | 主机 + AX650N PCIe 卡 |
| AX650 片上 | libax_engine / libax_sys | AX650N 板 |

主机侧为 numpy 实现,不依赖 torch / open3d。

## 目录结构

```
abot_axera/                 推理核心
  native_runner.py          NPU 链式推理(AXCL / 片上)
  runners.py                参考实现(pyaxengine / ONNX Runtime)
  backend.py                AbotRecon:图片序列 → 位姿 / 世界点 / 置信度
  pose_head.py              位姿头
  preprocess.py, csrc/      帧预处理
  pointcloud.py             点云
  progress.py               任务进度
service/                    FastAPI 服务 + viser 3D + 网页
scripts/                    验证脚本
```

## 模型与依赖

模型文件从 HuggingFace 下载:**https://huggingface.co/AXERA-TECH/ABot-Recon**

```bash
hf download AXERA-TECH/ABot-Recon --local-dir /path/to/ABot-Recon
```

| 内容 | 环境变量 | 说明 |
|---|---|---|
| `encoder_kitti02.axmodel` `decoder_step_kitti02.axmodel` `heads_kitti02.axmodel` | `ABOT_MODELS`(目录)、`ABOT_MODEL_SUFFIX`(默认 `_kitti02`) | pulsar2 编译的 AX650N 模型,共约 1.4 GB |
| `host_pose_head/pose_head.safetensors`、`host_pose_head/pose_head_config.json` | `ABOT_POSE_WEIGHTS`、`ABOT_POSE_CONFIG` | 主机侧位姿头 |

ONNX 模型运行不需要,仅 `scripts/validate_onnx.py` 评估量化误差时使用。

Python 依赖见 `requirements.txt`:推理只需 numpy、cffi、Pillow;服务另需 opencv、matplotlib、viser、fastapi、uvicorn。Axera 运行时与 pyaxengine 来自 SDK。预处理内核首次导入时自动编译(需 gcc),aarch64 已带预编译库。

## 运行

```bash
bash start_service.sh                                              # 前台
tmux new-session -d -s abot "bash start_service.sh > service_run.log 2>&1"   # 常驻
```

- 网页与 API:`http://<host>:8011`;viser 3D:`:8082`(内嵌在网页中)
- 网页流程:上传视频 → 选抽帧率 → 提交 → 进度条(帧 i/N、已用、预计剩余)→ 3D 查看 / 俯视图 / 下载 ply
- 模型在首次任务时加载,空闲 `MAP_IDLE_UNLOAD` 秒后卸载
- 任务卡片可直接播放原视频;同一视频只在 `jobs/_videos/` 存一份(按内容哈希),任务目录硬链接到它,删除任务后无引用的视频自动清除

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `ABOT_RUNNER` | `native` | `native` / `pyaxengine`(参考实现) |
| `ABOT_DEVICE` | `auto` | `axcl` / `ax650` / `auto` |
| `ABOT_DEVICE_ID` | `0` | AXCL 卡号(片上忽略) |
| `ABOT_MODELS` / `ABOT_MODEL_SUFFIX` | `/home/axera/ABot-Recon` / `_kitti02` | axmodel 目录与文件名后缀 |
| `ABOT_POSE_WEIGHTS` / `ABOT_POSE_CONFIG` | `$ABOT_DELIVERY/host_pose_head/...` | 位姿头权重与配置 |
| `MAP_PORT` / `VISER_PORT` | `8011` / `8082` | 服务端口 |
| `MAP_DATA` | `./jobs` | 任务产物目录 |
| `MAP_FPS` | `8` | 默认抽帧率(网页可改,1–15) |
| `MAP_IDLE_UNLOAD` | `100000` | 空闲多少秒卸载模型 |

### API

```
POST /jobs                file[, fps, ceiling_cut, ceiling_keep]  → {job_id, dedup, same_as}
GET  /jobs                任务列表;运行中的带 progress {phase, frame, total, elapsed_s, sec_per_frame, eta_s};same_as = 用同一视频的其他任务
GET  /jobs/{id}           任务状态、meta、产物
GET  /jobs/{id}/{file}    下载产物(floorplan.png / cloud.ply / recon.npz / view_*.png)
POST /jobs/{id}/view      载入 viser;POST /jobs/{id}/rerun 重跑;DELETE /jobs/{id} 删除
GET  /status              模型状态;POST /models/load|unload
```

## 资源消耗

模型输入 280×504(高×宽,即横版 16:9 画幅)。预处理把帧按宽度缩放到 504,再上下居中裁剪或填充到 280:横版 16:9 视频几乎不裁;竖版视频只保留中间约 1/3 画面,请使用横版视频。

### CMM(AX650N,axcl-smi 实测)

| 项 | CMM |
|---|---|
| encoder | 337 MiB |
| decoder_step | 2513 MiB |
| heads | 217 MiB |
| KV cache 缓冲(4 × 588 MB fp32) | 2243 MiB |
| 合计(常驻) | 5341 MiB |

AXCL 卡 CMM 为 7040 MiB。片上运行需同样约 5.3 GB CMM,板子 CMM 预留需 ≥ 6 GB。

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

估算:`总时长 ≈ 模型加载 + 抽帧 + N × 3.0 s + 0.13 s × N`,N ≈ 视频秒数 × 抽帧率。例:346 帧(43 s 视频,8 fps)任务 1084 s。`ABOT_RUNNER=pyaxengine` 时每帧约 9.5 s。

## Docker

每次推送 main 后 CI 自动构建三个镜像,放在固定的 release tag [`docker-latest`](https://github.com/AXERA-TECH/ABot-Recon/releases/tag/docker-latest)(每次覆盖),同时提供 `.tar` 和 `.tgz`:

| 文件 | 平台 | 用途 |
|---|---|---|
| `abot-recon-axcl-x86_64.tar` / `.tgz` | linux/amd64 | x86 主机 + AXCL 卡 |
| `abot-recon-axcl-aarch64.tar` / `.tgz` | linux/arm64 | aarch64 主机 + AXCL 卡 |
| `abot-recon-ax650-aarch64.tar` / `.tgz` | linux/arm64 | AX650 板片上 |

镜像内不含 Axera 运行时,宿主机需已安装 AXCL 驱动(或板子 BSP),运行时从宿主机映射进容器;`/models` 目录按 HuggingFace 仓库原样布局(含 `host_pose_head/`)。

```bash
# 载入镜像
docker load -i abot-recon-axcl-x86_64.tgz          # .tar 同样可以

# 模型
hf download AXERA-TECH/ABot-Recon --local-dir /path/to/ABot-Recon

# AXCL 卡(x86_64 / aarch64 主机)
docker run -d --name abot -p 8011:8011 -p 8082:8082 \
  --device /dev/axcl_host --device /dev/ax_mmb_dev --device /dev/msg_userdev \
  -v /usr/lib/axcl:/usr/lib/axcl:ro -v /usr/bin/axcl:/usr/bin/axcl:ro \
  -v /path/to/ABot-Recon:/models:ro -v $PWD/jobs:/data/jobs \
  -e ABOT_DEVICE_ID=0 abot-recon:axcl-x86_64

# AX650 板片上
docker run -d --name abot -p 8011:8011 -p 8082:8082 --privileged \
  -v /soc:/soc:ro -v /path/to/ABot-Recon:/models:ro -v $PWD/jobs:/data/jobs \
  abot-recon:ax650-aarch64
```

打开 `http://<host>:8011`。环境变量与上文相同(`-e` 传入);容器内默认 `ABOT_MODELS=/models`、`MAP_DATA=/data/jobs`。

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
