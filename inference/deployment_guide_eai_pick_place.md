# MotusV2 eai_pick_place 远端推理部署指南

## 架构总览

机器人端机器 **无本地 GPU**，因此采用 **Server/Client 分离架构**：

```
┌─────────────────────────────────┐         ZMQ (TCP)        ┌──────────────────────────────────┐
│  Client (机器人端, 无GPU)        │ ←──────────────────────→ │  Server (5090 GPU 服务器)         │
│                                 │                           │                                  │
│  1. 采集三视角图片 (ZMQ→相机)    │   Request:                │  4. 加载模型权重 (17GB)           │
│  2. 读取机器人关节 state         │    - first_frame (图片)   │     + Wan2.2 + Qwen3-VL + T5     │
│  3. 图片拼接 + state 预处理      │    - state (16维)         │  5. predict_action_chunk (GPU)    │
│     (CPU, 纯 numpy/opencv)      │    - instruction          │     返回 [64, 16] 动作序列        │
│  6. 插值 + 执行动作队列          │   Response:               │                                  │
│     (IK → ctrl_dual_arm)        │    - actions [64, 16]     │  显存: ~18.8GB (decode=OFF)       │
│  7. 夹爪控制                     │                           │  延迟: 5步~254ms / 10步~461ms    │
│                                 │                           │                                  │
│  unitree_lerobot (硬件接口)      │                           │  InferenceMotusV2 代码包          │
│  motus_inference_client.py       │                           │  motus_inference_server.py        │
└─────────────────────────────────┘                           └──────────────────────────────────┘
        │  DDS (unitree_sdk2py)
        ▼
┌──────────────────┐
│  G1 机器人本体    │
│  + 相机服务       │
│  192.168.123.164  │
└──────────────────┘
```

### 数据流

```
Client 每个控制周期 (30Hz):
  1. img_client.get_head_frame() / get_left_wrist_frame() / get_right_wrist_frame()
     → 三视角图片 (来自机器人相机服务, ZMQ SUB)
  2. arm_ctrl.get_current_dual_arm_q() → 14维关节角
  3. build_stitched_image(head, left_wrist, right_wrist) → 384×320×3 拼接图 (CPU)
  4. 构建 state: [L7, R7, LG, RG] → reorder → [L7, LG, R7, RG] (16维)
  5. ZMQ发送: {first_frame: 384×320×3 uint8, state: 16 float32, instruction: str}
  6. ZMQ接收: {actions: [64, 16] float32}
  7. interpolate_chunk(actions) → factor=1 时 no-op
  8. step() → (arm_action[14], left_grip, right_grip)
  9. arm_ik.solve_tau(arm_action) → tau
  10. arm_ctrl.ctrl_dual_arm(arm_action, tau) → 发送到机器人 (DDS)
  11. _write_gripper(left_grip, right_grip) → 夹爪 (DDS)
```

### 异步双缓冲

Client 使用 worker 线程发送推理请求，主线程持续执行动作队列：
- 队列长度 ≤ trigger(32) 时触发下一次推理
- 推理期间机器人持续执行上一个 chunk 的剩余动作
- 需要 chunk 执行时长 > 推理延迟 (64步@30Hz=2.13s >> 0.46s 推理)

---

## 需要传输的文件清单

### 1. Server 端 (5090 GPU 服务器)

#### 1.1 代码 — InferenceMotusV2 包 (已解压)

| 本地路径 | 服务器路径(建议) | 说明 |
|----------|------------------|------|
| `/home/ma-user/work/wx1513998/InferenceMotusV2/` | `/home/robo/InferenceMotusV2/` | 整个代码包，含 model 代码 + adapter + 脚本 |

**关键子目录**:
- `policy/MotusWanVlmDirectMaskMotion/` — 模型定义代码 (models/, bak/wan/, utils/)
- `scripts/motusv2_adapter/adapter.py` — MotusV2PolicyAdapter (服务端只用 predict_action_chunk)
- `scripts/motus_inference_server.py` — **新增** ZMQ 服务端
- `configs/eai_pick_place_wan_vlm_mask.yaml` — **新增** 推理配置 (路径已改为服务器端)
- `scripts/inference_motusv2_server.sh` — **新增** 服务端启动脚本

#### 1.2 模型权重 — 训练 checkpoint

| 本地路径 | 服务器路径(建议) | 大小 | 说明 |
|----------|------------------|------|------|
| `.../checkpoint_step_10000/pytorch_model/mp_rank_00_model_states.pt` | `/home/robo/pretrained_models/ckpts/eai_pick_place/mp_rank_00_model_states.pt` | 17GB | **唯一需要的权重文件** |

> **不需要传**: `config.json`, `pytorch_model_0.bin`, `*_optim_states.pt`, `random_states_*.pkl`, `zero_to_fp32.py`, `latest`

也可以直接传整个 `checkpoint_step_10000/` 目录，但只有 `pytorch_model/mp_rank_00_model_states.pt` 会被加载。

#### 1.3 预训练基座模型 (已传出)

| 本地路径 | 服务器路径(建议) | 大小 | 说明 |
|----------|------------------|------|------|
| `/home/ma-user/work/wx1513998/pretrained_models/Wan2.2-TI2V-5B` | `/home/robo/pretrained_models/Wan2.2-TI2V-5B` | ~20GB | WAN 2.2 视频生成基座 |
| `/home/ma-user/work/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct` | `/home/robo/pretrained_models/Qwen3-VL-2B-Instruct` | ~5GB | Qwen3-VL 语言模型 |

#### 1.4 T5 缓存

| 本地路径 | 服务器路径(建议) | 大小 | 说明 |
|----------|------------------|------|------|
| `.../pick_place_100_ewam/t5_cache/` | `/home/robo/pretrained_models/t5_cache_eai_pick_place/` | 8.1MB | UMT5-XXL 指令编码缓存 (单文件, 指令的 SHA1 哈希命名) |

> T5 缓存很小，但**必须传**。如果缺失，服务器会尝试加载 11GB 的 UMT5-XXL 编码器，可能 OOM。

#### 1.5 配置文件

| 本地路径 | 服务器路径(建议) | 说明 |
|----------|------------------|------|
| `MotusV21/MotusV2/configs/eai_pick_place_wan_vlm_mask.yaml` | `InferenceMotusV2/configs/eai_pick_place_wan_vlm_mask.yaml` | 训练配置，**需要修改路径**指向服务器端 |

**需要修改的路径字段** (在 YAML 中):
```yaml
dataset:
  t5_cache_dir: "/home/robo/pretrained_models/t5_cache_eai_pick_place"  # 服务器端 T5 缓存路径
  params:
    wan_path: "/home/robo/pretrained_models"  # 服务器端预训练模型根目录

model:
  wan:
    checkpoint_path: "/home/robo/pretrained_models/Wan2.2-TI2V-5B"
    config_path: "/home/robo/pretrained_models/Wan2.2-TI2V-5B"
    vae_path: "/home/robo/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  vlm:
    checkpoint_path: "/home/robo/pretrained_models/Qwen3-VL-2B-Instruct"
```

**不需要修改的字段** (这些与路径无关，必须保持与训练一致):
```yaml
common:
  action_dim: 16
  state_dim: 16
  num_video_frames: 8
  video_height: 384
  video_width: 320
  global_downsample_rate: 1
  video_action_freq_ratio: 8    # → action_chunk_size = 64

dataset:
  type: "g1d"
  stitch_mode: "aspect"         # 必须与训练一致
```

### 2. Client 端 (机器人端机器)

#### 2.1 代码

| 本地路径 | 机器人端路径(建议) | 说明 |
|----------|---------------------|------|
| `InferenceMotusV2/scripts/motus_inference_client.py` | `/home/robo/scripts/motus_inference_client.py` | **新增** ZMQ 客户端 |
| `InferenceMotusV2/scripts/inference_motusv2_client.sh` | `/home/robo/scripts/inference_motusv2_client.sh` | **新增** 客户端启动脚本 |
| `InferenceMotusV2/scripts/motusv2_adapter/adapter.py` | `/home/robo/scripts/motusv2_adapter/adapter.py` | 客户端只需要预处理方法 (stitching, state, interpolation) |
| `InferenceMotusV2/scripts/motusv2_adapter/__init__.py` | 同上 | |
| `InferenceMotusV2/scripts/motusv2_adapter/debug.py` | 同上 | |
| `InferenceMotusV2/policy/MotusWanVlmDirectMaskMotion/utils/image_utils.py` | `/home/robo/scripts/image_utils.py` | `resize_with_padding` 函数 |
| `InferenceMotusV2/configs/eai_pick_place_wan_vlm_mask.yaml` | `/home/robo/configs/eai_pick_place_wan_vlm_mask.yaml` | 配置文件 (客户端只需要 common + stitch_mode 字段) |

#### 2.2 外部依赖 (机器人端)

| 依赖 | 说明 |
|------|------|
| `unitree_lerobot` 仓库 | 硬件接口: ImageClient (ZMQ相机), G1_29_ArmController (DDS手臂), Dex1_1_Gripper_Controller (夹爪) |
| `unitree_sdk2py` | Unitree DDS SDK |
| `zmq` (pyzmq) | ZMQ 客户端通信 |
| `numpy`, `opencv-python`, `pyyaml` | 图像处理和配置读取 |

> 机器人端**不需要**: torch, transformers, WAN, Qwen3-VL, T5, DeepSpeed — 这些只在服务端

---

## 服务端配置详解 (config.yaml 字段对照)

| YAML 字段 | 训练端值 | 服务端值 | 说明 |
|-----------|---------|---------|------|
| `common.action_dim` | 16 | 16 | 动作维度 (qpos: 7+7+1+1) |
| `common.state_dim` | 16 | 16 | 状态维度 |
| `common.num_video_frames` | 8 | 8 | 视频帧数 |
| `common.video_height` | 384 | 384 | 拼接图高度 |
| `common.video_width` | 320 | 320 | 拼接图宽度 |
| `common.global_downsample_rate` | 1 | 1 | 下采样率 (决定 action_interp_factor) |
| `common.video_action_freq_ratio` | 8 | 8 | → chunk_size = 8×8 = 64 |
| `dataset.type` | "g1d" | "g1d" | 数据集类型 |
| `dataset.stitch_mode` | "aspect" | "aspect" | 图片拼接模式 (**必须与训练一致**) |
| `dataset.t5_cache_dir` | 训练机路径 | **服务器路径** | T5 缓存目录 |
| `dataset.params.wan_path` | 训练机路径 | **服务器路径** | 预训练模型根目录 |
| `model.wan.checkpoint_path` | 训练机路径 | **服务器路径** | WAN 2.2 路径 |
| `model.vlm.checkpoint_path` | 训练机路径 | **服务器路径** | Qwen3-VL 路径 |
| `model.inference.num_inference_timesteps` | 10 | 10 (或5) | 去噪步数 (5更快, 10更精细) |

---

## 启动顺序

### Step 1: 启动服务端 (5090 GPU 服务器)

```bash
# 在 5090 服务器上
cd /home/robo/InferenceMotusV2
bash scripts/inference_motusv2_server.sh
```

服务端会:
1. 加载模型权重 (~30秒)
2. 加载 Wan2.2 + Qwen3-VL + T5 缓存
3. 绑定 ZMQ REP socket 到 `tcp://0.0.0.0:5555`
4. 等待客户端请求

### Step 2: 启动客户端 (机器人端机器)

```bash
# 在机器人端机器上
cd /home/robo
bash scripts/inference_motusv2_client.sh
```

客户端会:
1. 连接相机服务 (ZMQ → 192.168.123.164)
2. 连接机器人手臂 (DDS)
3. 连接推理服务端 (ZMQ → 5090服务器IP:5555)
4. 等待用户输入 's' 开始
5. 开始 30Hz 控制循环

### Step 3: 开始任务

在客户端终端输入 `s` + 回车，机器人开始执行 pick_place 任务。

---

## 网络要求

| 连接 | 协议 | 延迟要求 | 带宽 |
|------|------|---------|------|
| Client → 相机服务 | ZMQ SUB | <5ms | ~10MB/s (3路视频) |
| Client → 机器人手臂 | DDS | <1ms | 极小 |
| Client → 推理服务端 | ZMQ REQ/REP | <10ms | ~1MB/request (图片) |
| 推理服务端 → Client | ZMQ REP/REQ | <10ms | ~4KB/response (动作) |

> **关键**: Client → 推理服务端的往返延迟 (RTT) 应 <100ms。推理本身 ~250-460ms，加上网络延迟后总延迟应 <600ms。一个 chunk 执行 2.13s (64步@30Hz)，远大于推理延迟，异步模式可保证连续运动。

---

## 调试

### 服务端日志
服务端会打印每次推理的耗时:
```
[server] predict: t5=0.5ms pil=0.3ms vlm=5.2ms forward=254.0ms total=260.0ms
```

### 客户端日志
客户端会打印每个控制周期的耗时:
```
[0|predict] image_net=7.2ms obs_preprocess=3.1ms server_rtt=280.5ms chunk=(64, 16) queue=64
[1|cached] loop=2.1ms get_q=0.5ms step=0.1ms ik=1.9ms ctrl=0.3ms grip=0.1ms
```

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 服务端 OOM | 显存不足 | 确认 `decode_video=False` (默认), 使用 5 步推理 |
| T5 cache miss | 缓存路径不对 | 检查 YAML 中 `t5_cache_dir` 指向正确 |
| 图片全黑 | stitch_mode 不匹配 | 确认 YAML 中 `stitch_mode: "aspect"` |
| 动作异常 | checkpoint 不匹配 | 确认 `mp_rank_00_model_states.pt` 来自正确的训练 run |
| 连接超时 | 网络不通 | 检查 5090 服务器防火墙放行 5555 端口 |
| 机器人不动 | DDS 接口不对 | 检查 `UNITREE_DDSINTERFACE` 环境变量 |
