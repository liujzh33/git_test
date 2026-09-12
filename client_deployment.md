# MotusV2 eai_pick_place — 客户端部署与启动说明

> 面向机器人端（无 GPU）操作人员。服务端已经在 5090 机器上跑起来了，本文只讲客户端要做什么。
>
> 最后更新：2026-09-12

---

## 1. 现在的状态

服务端**已经在运行**，不需要你在 5090 那边做任何事：

| 项 | 值 |
|---|---|
| 服务端地址 | `192.168.0.180:5555`（内网，不能直连） |
| GPU | GPU 6，占用 19.7GB / 32.6GB |
| 权重 | `/data/checkpoint/motusv2_20000/motusv2_20000/mp_rank_00_model_states.pt`（20000 step，17,049,838,933 字节） |
| 权重校验 | `missing=0, unexpected=0`（与模型定义完全匹配） |
| 去噪步数 | 10 步 |
| 动作块 | `(64, 16)` float32，@30Hz = 2.13 秒 |
| 推理延迟 | 稳态约 1.07 秒（首次请求含 CUDA warmup，约 1.9 秒） |
| 日志 | 5090 机器上 `/tmp/motus_server.log` |

服务端启动日志末尾长这样，看到最后一行就说明它在等请求：

```
[server] Checkpoint loaded from /data/checkpoint/motusv2_20000/motusv2_20000/mp_rank_00_model_states.pt: missing=0, unexpected=0
[server] T5 cache hit for the configured instruction -> UMT5-XXL encoder NOT loaded
[server] Model loaded in 122.0s. chunk_size=64, action_mode=qpos, stitch_mode=aspect
[server] Server listening on tcp://0.0.0.0:5555
```

---

## 2. 架构

机器人端没有 GPU，所以拆成服务端 / 客户端两半，中间走 ZMQ REQ/REP：

```
机器人端（无 GPU）                    SSH 隧道              5090 服务器（192.168.0.180）
┌────────────────────────┐                              ┌──────────────────────────┐
│ 1. 采三路相机图        │  ──── 请求 ────────────────>  │ 4. MotusV2 推理（GPU 6） │
│ 2. 读机械臂关节 state  │      first_frame 384x320x3   │    WAN 5B + Qwen3-VL 2B  │
│ 3. 拼图 + state 重排   │      state 16 维             │    + Action Expert       │
│                        │      instruction             │                          │
│ 6. 执行动作队列        │  <─── 响应 ────────────────   │ 5. 返回 (64, 16) 动作块  │
│    IK -> ctrl_dual_arm │      actions [64, 16]        │                          │
└────────────────────────┘                              └──────────────────────────┘
         │ DDS
         v
   G1 机器人 + 相机服务（192.168.123.164）
```

客户端**不需要** torch / transformers / WAN / Qwen3-VL / deepspeed，这些只在服务端。

---

## 3. 客户端依赖

| 依赖 | 说明 |
|---|---|
| `unitree_lerobot` | 硬件接口：ImageClient（ZMQ 相机）、G1_29_ArmController（DDS 手臂）、Dex1_1_Gripper_Controller（夹爪） |
| `unitree_sdk2py` | Unitree DDS SDK |
| `pyzmq` | 与推理服务端通信 |
| `numpy` / `opencv-python` / `pyyaml` / `pyarrow` | 图像处理、配置读取、读初始位姿 |

---

## 4. 传输代码

在**客户端机器**上执行，一条命令拉取 + 解压 + 校验：

```bash
scp -P 34134 -i /data2/liujingzhi/id_ed25519_5090 root@116.63.180.90:/home/work/git_test-main/motus_client_pkg.tar.gz . && tar xzf motus_client_pkg.tar.gz && md5sum motus_client_pkg.tar.gz && ls -R motus_client
```

md5 应该是 `e426c928fad43f2b06e89a7fb821a243`。

> `scp` 用大写 `-P` 指定端口，`ssh` 用小写 `-p`，这是最容易踩的坑。密钥路径按你机器上的实际位置改。

包里有四个文件：

| 文件 | 作用 |
|---|---|
| `scripts/motus_inference_client.py` | 客户端主程序 |
| `scripts/inference_motusv2_client.sh` | 启动脚本，默认已指向隧道 `127.0.0.1:15555` |
| `scripts/smoke_test_server.py` | 隧道连通性测试，不碰机器人 |
| `configs/eai_pick_place_wan_vlm_mask.yaml` | 配置，客户端只读 `common` 和 `dataset.stitch_mode` |

---

## 5. 启动（三个终端）

### 终端 A：SSH 隧道（必须，保持运行）

5090 机器不能直接对外开端口，所以要把它的 5555 映射到客户端本地的 15555：

```bash
ssh -p 34134 -i /data2/liujingzhi/id_ed25519_5090 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -N -L 127.0.0.1:15555:192.168.0.180:5555 \
  root@116.63.180.90
```

`-N` 表示只建隧道不开 shell，命令会一直挂着不返回，这是正常的。`ServerAliveInterval` 防止空闲断连。

### 终端 B：连通性测试（建议先跑，不碰机器人）

```bash
python motus_client/scripts/smoke_test_server.py --host 127.0.0.1 --port 15555
```

正常输出：

```
connected to tcp://127.0.0.1:15555
[0] rtt= 1948.9ms  server_predict= 1947.0ms  chunk=(64, 16) float32  range=[-0.209, 4.469]
[1] rtt= 1279.2ms  server_predict= 1277.8ms  chunk=(64, 16) float32  range=[-0.173, 4.344]
[2] rtt= 1075.1ms  server_predict= 1074.2ms  chunk=(64, 16) float32  range=[-0.184, 4.250]
```

看到 `chunk=(64, 16)` 就说明隧道通了、服务端能正常响应。这一步失败了也不会动机器人，可以放心试。

### 终端 C：启动客户端

```bash
cd motus_client && \
LEROBOT_ROOT=/home/robo/codes/unitree_lerobot/ \
UNITREE_DDSINTERFACE=eth0 \
IMAGE_HOST=192.168.123.164 \
bash scripts/inference_motusv2_client.sh
```

三个环境变量按实际情况改：

- `LEROBOT_ROOT` — unitree_lerobot 仓库路径
- `UNITREE_DDSINTERFACE` — 机器人那张网卡，用 `ip link` 查
- `IMAGE_HOST` — 相机服务地址

`SERVER_HOST=127.0.0.1` 和 `SERVER_PORT=15555` 脚本里已经是默认值，不用传。

启动后会打印配置，然后等你输入 `s` + 回车才开始 30Hz 控制循环。

---

## 6. 任务指令

```
Pick up the marker and the plush toy and place them into the right basket, then pick up the empty water bottle and the draft paper and place them into the left basket
```

**这条字符串必须一字不差**（首字母大写 `Pick`，结尾不带句号）。服务端的 T5 文本编码走磁盘缓存，文件名是 `sha1(instruction).pt`，改一个字符就 cache miss，服务端会去加载 11GB 的 UMT5-XXL 编码器，可能直接 OOM。

客户端脚本里已经写死了正确的字符串，除非你要换任务，否则不要改 `INSTRUCTION`。

---

## 7. 动作维度与顺序

服务端返回 `(64, 16)` float32：64 个时间步 × 16 维，@30Hz 覆盖 2.13 秒。

16 维在网络传输上是**左右交错**排列的：

| 下标 | 含义 |
|---|---|
| 0–6 | 左臂 7 关节 |
| 7 | **左夹爪** |
| 8–14 | 右臂 7 关节 |
| 15 | **右夹爪** |

注意这和机器人硬件的原始顺序不同。硬件是 `[左臂7, 右臂7, 左夹爪, 右夹爪]`，客户端上行时用 `_REORDER_FROM_RAW = [0,1,2,3,4,5,6,14,7,8,9,10,11,12,13,15]` 重排，下行用 `_from_arm_interleaved` 排回去。这部分客户端已经处理好了，只有你要自己解析动作时才需要关心。

数值是**绝对 qpos**（关节为弧度，夹爪为连续值），**没有做归一化**，拿到就能直接送 IK。

---

## 8. 已知情况：动作会不连续

这台机器上实测推理延迟比原部署文档里写的慢一倍多（没装 flash_attn，torch.compile 也是关的）：

| 去噪步数 | 实测延迟 | 文档标称 |
|---|---|---|
| 5 步 | ~560ms | ~254ms |
| 10 步 | ~1050ms | ~461ms |

异步双缓冲要连续运动，需要同时满足 `ASYNC_TRIGGER >= I` 和 `chunk >= ASYNC_TRIGGER + I`（`I` 是推理延迟折算成的执行步数）。10 步时 `I ≈ 33` 步，而 chunk 只有 64 步，`33 <= trigger <= 31` 无解，所以**机器人会在两个动作块之间短暂停顿**。

目前配的就是 10 步，因为当前目标是看任务成功率，画面连贯性不重要，10 步生成质量更高。

如果后面想要平滑运动，让 5090 那边加个环境变量重启服务端即可（5 步时 `I ≈ 18` 步，`18 <= trigger <= 46`，默认的 `trigger=32` 落在区间内）：

```bash
# 在 5090 机器上
cd /home/work/git_test-main/InferenceMotusV2 && \
NUM_INFERENCE_STEPS=5 setsid nohup bash scripts/inference_motusv2_server.sh > /tmp/motus_server.log 2>&1 < /dev/null &
```

客户端那边对应把 `NUM_INFERENCE_STEPS=5` 也加上（两边要一致）。

---

## 9. 排查

| 现象 | 原因 | 处理 |
|---|---|---|
| smoke test 卡住不返回 | 隧道没建或断了 | 检查终端 A 是否还活着；`ss -lnt \| grep 15555` 看本地端口有没有监听 |
| `Connection refused` | 隧道在但服务端挂了 | 登 5090 看 `tail /tmp/motus_server.log`，确认最后一行是 `Server listening` |
| 隧道频繁断开 | 空闲超时 | 已加 `ServerAliveInterval=30`；还断就把间隔调小到 15 |
| 服务端日志出现 `T5 cache miss` | instruction 被改过 | 改回第 6 节那条原文，一个字符都不能差 |
| 图片全黑 / 动作乱飞 | `stitch_mode` 不匹配 | 确认 yaml 里是 `stitch_mode: "aspect"`，且用的是随包发的那份 yaml |
| 机器人不动 | DDS 网卡不对 | `ip link` 查网卡名，改 `UNITREE_DDSINTERFACE` |
| 相机取不到图 | 相机服务地址不对 | 改 `IMAGE_HOST` |

客户端正常运行时的日志长这样：

```
[0|predict] image_net=7.2ms obs_preprocess=3.1ms server_rtt=1075.1ms chunk=(64, 16) queue=64
[1|cached]  loop=2.1ms get_q=0.5ms step=0.1ms ik=1.9ms ctrl=0.3ms grip=0.1ms
```

`predict` 行是发生了一次远程推理，`cached` 行是在消费本地动作队列。
