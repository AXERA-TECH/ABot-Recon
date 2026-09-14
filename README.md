# ABot-Recon on Axera NPU

把 [ABot-Recon](https://huggingface.co/acvlab/ABot-Recon)(流式前馈三维重建:视频 → 相机位姿 + 世界坐标点图 + 置信度)
跑在 Axera NPU 上,并带一个建图服务(上传视频 → 点云 / 户型俯视图 / 3D 网页)。

支持两种运行形态,**同一套代码**:

| 形态 | 运行时 | 典型机器 |
|---|---|---|
| **AXCL** PCIe 卡 | `libaxcl_rt`(axclrt API) | x86 主机插 AX650N 卡 |
| **AX650 片上** | `libax_engine` + `libax_sys`(AX_ENGINE / CMM) | AX650N 开发板 |

整个仓库只依赖 numpy + cffi + pyaxengine(借它的 cffi 声明)+ Pillow:主机侧的预处理、位姿头、点云都是 numpy
(缩放这一步是一个 60 行的 C 内核,cffi 调用),**没有 torch / torchvision / open3d / safetensors**,裸 AX650 板子也能跑。
和原来 torch 路径的结果比:预处理逐位一致,位姿差 ~1e-7,点云点数完全相同。

---

## 目录结构

```
.
├── README.md  requirements.txt  start_service.sh
├── abot_axera/                # Axera NPU 后端(本仓库的核心)
│   ├── native_runner.py       #   原生 runner:自管设备内存,KV cache 常驻设备(AXCL + AX650 片上)
│   ├── runners.py             #   参考 runner:pyaxengine 逐模型 InferenceSession / ONNX Runtime
│   ├── backend.py             #   AbotRecon:图片序列 → 位姿 / 世界点 / 置信度
│   ├── pose_head.py           #   主机侧 AdjacentPoseHead(numpy 移植,自带 safetensors 解析)
│   ├── preprocess.py          #   抗锯齿 bicubic 缩放 + 裁剪/填充(C 内核逐位等于 torchvision;numpy 兜底)
│   ├── csrc/resize_aa.c       #   缩放内核:照抄 PyTorch _upsample_bicubic2d_aa,首次使用时 gcc 编译,cffi 调用
│   ├── prebuilt/              #   没有编译器的板子用的预编译 .so(aarch64)
│   ├── pointcloud.py          #   体素下采样 + 二进制 PLY 读写
│   └── progress.py            #   逐帧进度 + 预估剩余时间(网页进度条数据源)
├── service/
│   ├── service.py             # FastAPI:任务队列 / 模型热加载 / viser 3D
│   ├── mapping_pipeline.py    # 视频 → 抽帧 → 推理 → 点云 / 户型图 / 渲染
│   └── index.html             # 网页面板
└── scripts/
    ├── validate_native.py     # 原生 runner vs pyaxengine 逐位比对 + 计时
    ├── validate_onnx.py       # NPU 链路 vs ONNX golden(cos / relL2)
    ├── validate_torchfree.py  # numpy 预处理/位姿头/点云 vs torch/torchvision/open3d 原版(一次性)
    ├── dump_feats.py          # 导出一段视频的 camera_features 等做对照数据
    ├── onchip_check.py        # 片上 AX650 板子验证(encoder + heads)
    └── http_smoke.sh          # HTTP 端到端冒烟
```

---

## 模型与依赖(不在仓库里)

| 内容 | 环境变量 | 说明 |
|---|---|---|
| `encoder_*.axmodel` `decoder_step_*.axmodel` `heads_*.axmodel` | `ABOT_MODELS` + `ABOT_MODEL_SUFFIX`(默认 `_kitti02`) | pulsar2 编译的 AX650N 模型,三个共约 1.4 GB |
| `host_pose_head/pose_head.safetensors` + `pose_head_config.json` | `ABOT_POSE_WEIGHTS` / `ABOT_POSE_CONFIG`(或 `ABOT_DELIVERY` 交付包根目录) | 主机侧位姿头 |
| `models/onnx/*.onnx`(可选) | `ABOT_DELIVERY` | 仅 `validate_onnx.py` 用 |

Python 依赖见 `requirements.txt`(numpy / cffi / Pillow,服务再加 opencv / matplotlib / viser / fastapi);Axera 运行时和 pyaxengine 来自 SDK,不走 pip。
缩放内核需要 `gcc`(首次导入时编译到 `abot_axera/_build/`);没有编译器时先找 `abot_axera/prebuilt/` 里的预编译 .so(aarch64 已带),再退回 numpy 实现(权重精确,只差 fp32 累加顺序,~4e-7),`ABOT_RESIZE_NUMPY=1` 可强制。
同一份 C 源有两种循环布局:x86 用和 aten 一样的逐元素标量写法(`-O3 -mavx2 -mfma`,GCC 生成的顺序归约和 PyTorch 自己的 AVX2 内核一致,所以逐位相同);aarch64 用 `-DABOT_RESIZE_FAST` 的 RGB0 四通道交织布局,两个 pass 都是连续 NEON 乘加,比标量快 3 倍,和 x86 结果差 ≤1 ulp(板上本来就没有 torch 参照)。

---

## 推理流程

三个 axmodel 按帧串行,一帧一轮:

```
image[1,3,280,504] ─encoder─▶ patch_tokens[1,720,1024]
                                     │
   past_key/past_value[1,18,16,11,725,64] + past_valid + frame_index
                                     ▼
                              decoder_step ─▶ fused_hidden[1,725,2048] + present_key/value/valid
                                     │                    (present 直接作为下一帧的 past)
                                     ▼
                                   heads ─▶ local_points[280,504,3]  camera_features[725,512]  confidence[280,504,1]
```

每帧的 camera_features 立刻喂给主机侧 `AdjacentPoseHead`(numpy)串成 c2w 位姿(第 0 帧为单位阵);
跑完后 `world = R·local + t`,置信度过 sigmoid;再由 `mapping_pipeline` 上色、地面对齐、裁剪、体素下采样、写 ply、画图。

### 原生 runner 为什么快

pyaxengine 的 `InferenceSession.run()` 每次调用把**所有**输入拷上设备、**所有**输出拷回主机。
对 decoder_step 来说就是两个 588 MB 的 KV 张量进+出,每帧 2.35 GB 过 PCIe,占了 9.5 s/帧里的 6.5 s,
而 NPU 本身只要 2.6 s。`native_runner.NativeChainRunner`:

- `past_*` ↔ `present_*` 用**两组设备缓冲乒乓**:每帧把 present 输出缓冲重新绑成下一帧的 past 输入,指针交换,零拷贝;
- encoder→decoder 的 `patch_tokens`、decoder→heads 的 `fused_hidden` 是共享设备缓冲,不回主机;
- 每帧只上传 1.7 MB 图像、下载 ~3.8 MB heads 输出;
- 设备操作抽成一个很薄的接口(`malloc / memset0 / h2d / d2h / bind_in / bind_out / run`),两个实现:
  `_AxclDevice`(axclrtMalloc / axclrtMemcpy / axclrtEngineSetInputBufferByIndex / axclrtEngineExecute)和
  `_AxeDevice`(AX_SYS_MemAllocCached + AX_ENGINE_RunSyncV2,主机写后 flush、读前 invalidate,设备内部流转的缓冲不做 cache 维护)。

### 主机侧(numpy)对照 torch 原版

| 环节 | 对照 | 结果 |
|---|---|---|
| 预处理缩放(C 内核,uint8 直入) | torchvision to_tensor + bicubic antialias resize | 逐位一致(9 张 720p/1080p 图 0 像素差) |
| 预处理缩放(numpy 兜底) | 同上 | 最大 3.6e-7 |
| 位姿头 346 帧 | torch AdjacentPoseHead | 最大 4e-6(轨迹跨度 2.7) |
| 世界点变换 / sigmoid | torch | 1e-7 |
| 体素下采样 | open3d | 点数完全相同 |
| 端到端 12 帧任务 | torch 路径服务 | 原始点/云点完全相同,位姿 2.4e-7 |
| 端到端 346 帧任务 | torch 路径服务 | 见下表 |

`scripts/validate_torchfree.py` 可复现上表(需要临时装 torch/torchvision/open3d)。

### 主机侧各环节耗时

| 环节 | x86(dell,24 核) | AX650 板(A55×8,负载 5) |
|---|---|---|
| JPEG 解码(PIL) | ~1 ms/帧 | ~35 ms/帧(CPU 解码是板上瓶颈,后续换 ax-pipeline 硬解) |
| 缩放(C 内核,含 uint8 转换) | ~4 ms/帧 | ~10 ms/帧(交织布局;标量布局 32 ms) |
| 缩放 numpy 兜底 | ~60 ms/帧 | 太慢,板上请用 prebuilt |
| 位姿头(numpy) | ~10 ms/帧 | ~64 ms/帧 |
| 体素下采样 6.7M 点(1 次分配 + 2 组均值) | 2.3 s(open3d 6 s) | ~19 s |

板上 NPU:encoder 191 ms、heads 130 ms(和卡上一致,输出逐位相同);decoder_step 需 ≥6 GB CMM,常见板跑不下。

### 性能(AX650N,AXCL 卡,2026-09)

| 阶段 | 原生 runner | pyaxengine | 纯 NPU(axcl_run_model) |
|---|---|---|---|
| encoder | 0.20 s | 0.20 s | 0.19 s |
| decoder_step | 2.6 s | 9.1 s | 2.63 s |
| heads | 0.16 s | 0.16 s | 0.13 s |
| **每帧** | **~3.0 s** | ~9.5 s | |

时间估算:`总时长 ≈ 模型加载(首次 ~23 s)+ 抽帧 + N × 3.0 s + 后处理(≈ 0.13 s × N + 5 s)`,
N ≈ 视频秒数 × 采样 fps(默认 8)。一分钟视频约 25 分钟。网页进度条会按实测的每帧耗时给出预计剩余时间。

### 内存

- AXCL 卡:三个模型 ~3.2 GB CMM + KV 乒乓 2.35 GB ≈ 5.6 GB。
- AX650 片上同样需要 ≈ 5.6 GB CMM。常见开发板 CMM 只预留 4–4.6 GB,跑不下 decoder_step;
  encoder + heads(~1 GB)已在板上验证与卡上逐位一致(`scripts/onchip_check.py`)。

---

## 起服务

```bash
bash start_service.sh                      # 默认:ABOT_RUNNER=native, ABOT_DEVICE=auto
tmux new-session -d -s abot "bash start_service.sh > service_run.log 2>&1"   # 常驻
```

- 面板 + API:`http://<host>:8011`;viser 3D:`:8082`(面板内 iframe 内嵌)
- 网页:上传视频 → 选 fps → 提交 → 进度条(帧 i/N · 已用 · 预计剩余)→ 完成后「在3D查看」/「俯视图」/下载 ply
- 模型首次提交任务时热加载,空闲 `MAP_IDLE_UNLOAD` 秒后自动卸载

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `ABOT_RUNNER` | `native` | `native` KV 常驻设备;`pyaxengine` 逐模型 InferenceSession(参考/兜底) |
| `ABOT_DEVICE` | `auto` | 原生 runner 的运行时:`axcl` / `ax650` / `auto`(按本机存在的库选) |
| `ABOT_DEVICE_ID` | `0`(start_service.sh 里是 `6`) | AXCL 卡号;片上忽略 |
| `ABOT_MODELS` / `ABOT_MODEL_SUFFIX` | `/home/axera/ABot-Recon` / `_kitti02` | axmodel 目录和文件名后缀 |
| `ABOT_DELIVERY` | 交付包路径 | `host_pose_head/` 所在目录;或直接给 `ABOT_POSE_WEIGHTS` / `ABOT_POSE_CONFIG` |
| `MAP_PORT` / `VISER_PORT` | `8011` / `8082` | 服务端口 |
| `MAP_DATA` | `./jobs` | 任务产物目录 |
| `MAP_FPS` | `8` | 默认抽帧率(网页可改,1–15) |
| `MAP_IDLE_UNLOAD` | `100000` | 空闲多少秒卸载模型 |

### API

```
POST /jobs        file[,fps,ceiling_cut,ceiling_keep]   → {job_id}
GET  /jobs        所有任务;正在跑的带 progress {phase, frame, total, elapsed_s, sec_per_frame, eta_s}
GET  /jobs/{id}   任务状态 + meta + products
GET  /jobs/{id}/{file}   下载产物(floorplan.png / cloud.ply / recon.npz / view_*.png)
POST /jobs/{id}/view     把该任务点云载入 viser
POST /jobs/{id}/rerun    重跑;DELETE /jobs/{id} 删除
GET  /status      模型状态;POST /models/load|unload
```

---

## 验证

```bash
# 原生 runner vs pyaxengine:12 帧输出 + 末帧 KV 逐位比对(两次分开跑,各占 ~5.6 GB 卡内存)
python scripts/validate_native.py --runner pyaxengine --dev 7 --dump /tmp/ref.npz
python scripts/validate_native.py --runner native     --dev 7 --dump /tmp/nat.npz
python scripts/validate_native.py --compare /tmp/ref.npz /tmp/nat.npz     # PASS, 全 0 diff, x3.2

# NPU 链路 vs ONNX golden
python scripts/validate_onnx.py testdata/frames12 12

# numpy 主机侧 vs torch/torchvision/open3d 原版(一次性,需要装 torch)
python scripts/dump_feats.py <video.mp4> testdata/ref_2915 8          # 卡上导出 camera_features 等
python scripts/validate_torchfree.py --frames testdata/frames12 --feats testdata/ref_2915.npz

# 片上 AX650 板子(只需 python3 + numpy + cffi + pyaxengine)
python scripts/onchip_check.py --make-ref ref.npz            # 在卡主机上生成参考
python3 scripts/onchip_check.py /path/to/axmodels ref.npz    # 在板子上比对(encoder + heads)
```

---

## 已知限制

- **无回环闭合**(上游需要 SALAD / DINOv2 + CUDA)。长序列 revisit 的全局漂移不纠正,靠 `mapping_pipeline` 的相机轨迹 bbox 裁剪兜底。
- **present_valid 封顶 8**(应为 11):已知 axmodel 问题;短序列几乎无感,长序列有轻微漂移。
- 片上 AX650 需要 ≥ 6 GB CMM 才能跑完整链路(见「内存」)。
- 服务单任务串行。
