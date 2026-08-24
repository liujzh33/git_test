测 8080，走 SSH 隧道

服务器端：

cd /home/work/liujingzhi/websocket-test
python3 server_action.py --host 127.0.0.1 --port 8080

客户端机器建立隧道：

ssh -p 34134 -i /data2/liujingzhi/id_ed25519_5090 -N -L 18080:127.0.0.1:8080 root@116.63.180.90

客户端先只检查端口：

python3 /data2/liujingzhi/client_obs_action.py 127.0.0.1 --port 18080 --check-only

正式传输：

python3 /data2/liujingzhi/client_obs_action.py 127.0.0.1 --port 18080 --hz 10



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \                                                  
        --nproc_per_node=4 \                                                                                                                                                                                                                   
        --master_port=28200 \                                                                                                                                                                                                                  
        train/train_wan_vlm_mask.py \                                                                                                                                                                                                          
        --deepspeed configs/zero1.json \                                                                                                                                                                                                       
        --config configs/xinghaitu_wan_vlm_mask.yaml



 1. 数据格式与维度布局（部署最先要对齐的）

  1.1 目录结构（/cache/wx1513998/data/xinghaitu/X1-Robot-motus）

  episode_XXXXXX/
  ├── videos/episode_XXXXXX.mp4   # T 型拼接视频 320(W)×360(H), RGB, 10fps
  ├── qpos/episode_XXXXXX.pt      # [T, 40] float32, 绝对关节角(弧度)
  ├── metas/episode_XXXXXX.txt    # 完整 prompt 文本(含 scene prefix)
  └── umt5_wan/episode_XXXXXX.pt  # 预编码 T5 embedding, list[1] of [seq_len, 4096]

  1.2 qpos 40 维布局（必须严格对齐）

  [0:6]   base (steer×3 + wheel×3) — 全 0
  [6:10]  torso (4 关节)
  [10:17] left_arm state (7)
  [17:24] right_arm state (7)
  [24:31] left_arm action (7)  = state[10:17] 的目标
  [31:38] right_arm action (7) = state[17:24] 的目标
  [38]    left_gripper
  [39]    right_gripper

  1.3 训练用的 20 维 qpos_indices

  [24,25,26,27,28,29,30,  # left_arm(7)
   38,                     # L_gripper(1)
   31,32,33,34,35,36,37,  # right_arm(7)
   39,                     # R_gripper(1)
   6,7,8,9]               # torso(4)
  输出顺序固定为：[left_arm(7), L_gripper(1), right_arm(7), R_gripper(1), torso(4)]，这跟 data/utils/stat.json 里 xinghaitu 条目的 layout 一致。state 和 action 使用同一套 indices（abs
  位置控制，state=当前帧，action=未来帧的目标）。部署时模型输出的 20 维 action 必须按这个顺序解释，下发时再映射回机器人各自的关节。

  1.4 action chunk 时序

  - num_video_frames=8, video_action_freq_ratio=2 → action_chunk_size=16
  - global_downsample_rate=1 → 16 个物理帧，10fps 下覆盖 1.6s
  - 采样逻辑（_calculate_sampling_indices）：随机选 condition_frame_idx，action_indices[i] = condition + (i+1)*1，video_indices[i] = action_indices[(i+1)*2 - 1]（即第 1,3,5,...,15 个 action 帧作为视频帧）

  ---
  2. 归一化 —— 关键：action 不做归一化

  2.1 训练侧

  - XinghaituMotusDataset.__getitem__ 里完全没有调用 normalize_actions。虽然 import 了 data.utils.norm，但 __init__ 接收 embodiment_type="xinghaitu" 参数后从未使用（没有加载 stat.json）。
  - train/train_wan_vlm_mask.py 的训练循环也没有对 action 做任何归一化，直接 action_sequence = batch['action_sequence'] 送进 training_step。
  - 模型 motus_wan_vlm_direct_mask.py 里对 action 的处理只有 flow-matching 加噪：noisy_actions = actions * (1 - sigma) + noise * sigma，没有 min-max 缩放。

  结论：模型直接在物理关节角空间（弧度）学习。 stat.json 里的 xinghaitu min/max 只是统计值，训练时未使用。

  2.2 部署侧

  inference/real_world/Motus/server_vlm_mask.py 第 418-422 行：
  # Denormalize if required (Dobot uses [0,1] normalized actions during training)
  # if SERVER_STATE.action_denorm_required:
  #     action_min = torch.from_numpy(SERVER_STATE.action_min).unsqueeze(0)
  #     action_max = torch.from_numpy(SERVER_STATE.action_max).unsqueeze(0)
  #     predicted_actions_cpu = predicted_actions_cpu * (action_max - action_min) + action_min
  反归一化是注释掉的——因为星海图训练时没归一化，模型输出的就是原始弧度，直接下发即可。DATASET_ACTION_STATS 里只有 dobot 的条目，没有 xinghaitu。

  2.3 图像归一化

  - 数据集 load_video_frames：decord 读出 uint8 RGB → permute(0,3,1,2).float() / 255.0 → [T, C, H, W]，范围 [0,1]
  - 模型 training_step：first_frame * 2.0 - 1.0 → [-1, 1] 送入 WAN VAE
  - 推理 image_to_tensor：np.array(image).astype(np.float32) / 255.0 → permute(2,0,1) → [0,1]，模型内部再 *2-1

  ---
  3. RGB 通道顺序（部署易踩坑）

  3.1 训练

  - data/utils/image_utils.py 用 decord VideoReader.get_batch()，返回 RGB（HWC），代码注释明确写了 Color: decord returns frames in RGB with HWC layout。
  - tensor_to_pil 按 RGB 处理，Image.fromarray(image_np, mode='RGB')。
  - 所以训练视频 mp4 里的像素被当作 RGB 顺序送入模型。

  3.2 部署（服务端）

  - server_vlm_mask.py 所有图像入口都强制 .convert("RGB")（PIL），decode_base64_image、compose_multiview_image、image_to_tensor 都是 RGB。
  - compose_multiview_image：top=images[0]（头相机），bottom=[images[1]（左腕）, images[2]（右腕）]，np.concatenate 拼成 T 型。三路相机顺序必须是 [head, left_wrist, right_wrist]。

  3.3 R1-deploy 客户端

  - ros2_robot.py 的 decode_image_from_compressed：cv2.imdecode 默认返回 BGR，然后 cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 转成 RGB，再 astype(float32)/255.0，transpose(2,0,1) → CHW [0,1]。
  - 通过 websocket_policy_client 用 msgpack 发送时就是 RGB CHW [0,1]。服务端收到后需保持 RGB，不要再次 BGR↔RGB 转换。

  结论：整条链路都是 RGB。 部署时如果用 cv2 读图，必须 cvtColor(BGR2RGB)；如果用 PIL，必须 .convert("RGB")。

  ---
  4. T5 embedding（部署需要挂服务端实时编码的部分）

  4.1 模型与编码参数

  - 编码器：WAN2.2 自带的 umt5-xxl enc（Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth + google/umt5-xxl tokenizer）
  - 类：bak/wan/modules/t5.py::T5EncoderModel
  - text_len=512（max sequence length，padding 上限）
  - dtype=torch.bfloat16
  - tokenizer：HuggingfaceTokenizer(name='google/umt5-xxl', seq_len=512, clean='whitespace')，add_special_tokens=True，return_mask=True
  - 编码后输出维度 4096（umt5-xxl hidden size），实际序列长度按 token 数截断（如 episode_000000 是 77 token → [77, 4096]）

  4.2 预编码流程（离线）

  t5_encode_multigpu.py 或 convert_robotwin_pi0_to_motus.py 里：
  encoder = T5EncoderModel(text_len=512, dtype=bf16, device='cuda',
      checkpoint_path=".../models_t5_umt5-xxl-enc-bf16.pth",
      tokenizer_path=".../google/umt5-xxl")
  # 读 metas/episode_XXXXXX.txt（可多行，每行一个 prompt）
  prompts = [line for line in content.split("\n") if line.strip()]
  encoded = encoder(prompts, device)  # 返回 list，每个 [seq_len_i, 4096]
  torch.save([enc.cpu() for enc in encoded], "umt5_wan/episode_XXXXXX.pt")
  星海图每个 episode 的 metas/episode_XXXXXX.txt 只有一行，所以 umt5_wan/*.pt 是 list(len=1) of [seq_len, 4096]。

  4.3 训练时的 T5 加载

  XinghaituMotusDataset._load_language_embedding：
  embedding_data = torch.load(lang_path)  # list[1] of [seq, 4096]
  if isinstance(embedding_data, list):
      selected_idx = random.randint(0, len(embedding_data)-1)  # 多 prompt 随机选一个
      embeddings = embedding_data[selected_idx]
  if embeddings.dim() == 3:
      embeddings = embeddings.squeeze(0)
  return embeddings, selected_idx  # [seq, 4096]
  然后 collate_fn 把 batch 内多个 embedding pad 到等长（_process_language_embeddings_batch）。

  4.4 模型里 T5 的使用（VideoModule.preprocess_t5_embeddings）

  text_len = 512
  # 对每个样本：若 seq < 512 则 zero-pad 到 512，若 > 512 则截断
  padded = torch.cat([emb, emb.new_zeros(512 - emb.shape[0], emb.shape[1])])
  t5_context_raw = torch.stack(padded_embeddings, dim=0)  # [B, 512, 4096]
  # 通过 WAN 自带的 text_embedding 层 4096 -> 3072
  t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)  # [B, 512, 3072]
  这个 t5_context 在每个 WAN transformer block 的 cross_attn 里作为 K/V（wan_layer.cross_attn(norm3(video_tokens), t5_context, context_lens=None)），即 T5 只作用于 WAN 视频分支，不直接进 action_expert 或 VLM。

  4.5 部署时的 T5（关键）

  推理服务端 server_vlm_mask.py 目前的逻辑是加载预编码 .pt 文件，而不是实时编码：
  # 优先级：
  # 1. request.t5_embeddings_path  → 直接 load_t5_embeddings(path)
  # 2. auto_find_t5_embeddings + t5_embeddings_dir → 用 md5(instruction) 作为文件名查找
  # 3. 都没有 → 报错 "No T5 embeddings provided"
  load_t5_embeddings 支持 [seq, 4096] tensor 或 list of tensors。

  部署要挂实时编码服务端时，你需要：
  1. 加载 T5EncoderModel(text_len=512, dtype=bf16, checkpoint_path=..., tokenizer_path=...)（WAN 仓库里的 umt5-xxl）；
  2. 收到请求时用 encoder([instruction_text], device) 编码，取 [0] 得到 [seq, 4096]；
  3. 传给 inference_step 的 language_embeddings=[emb]（list 包一个 tensor）。
  4. instruction 文本必须和训练时一致：meta 文件里是 "The whole scene is in a realistic, industrial art style with three views: a fixed front camera, a movable left hand camera, and a movable right hand camera. The huawei hdas dual-arm 
  robot is currently performing the following task: <task>"。这个 prefix 必须保留，否则 T5 embedding 分布不匹配。

  ▎ 注意：server_vlm_mask.py 里的 DEFAULT_SCENE_PREFIX（"The whole scene is in a realistic environment..."）是给 dobot/franka 用的，不是星海图的 prefix。星海图用的是 "The whole scene is in a realistic, industrial art style with three 
  ▎ views: ..."（见 meta 文件和 data/robotwin2/robotwin_agilex_dataset.py:585）。

  ---
  5. VLM（Qwen3-VL-2B）输入构建

  5.1 训练侧（utils/vlm_utils.py::preprocess_vlm_messages）

  messages = [{"role":"user", "content":[
      {"type":"image", "image": first_frame_pil},   # 先图
      {"type":"text", "text": text_instruction}     # 后文
  ]}]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  image_inputs, video_inputs = process_vision_info(messages)
  inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt")
  - VLM 看到的是 first_frame（T 型拼接图）+ 完整 instruction 文本
  - add_generation_prompt=True（训练时就加 generation prompt，这有点特殊）
  - VLM 是可训练的（model.vlm.frozen: false）

  5.2 部署侧（server_vlm_mask.py::build_vlm_inputs）

  完全一致的流程，messages 顺序也是 image→text，add_generation_prompt=True。输入图也是 T 型拼接后的 PIL 图。

  5.3 VLM 在模型里的作用

  - qwen3_module.extract_per_layer_features(vlm_inputs) 抽取 30 层特征（vlm_dim=2048）
  - 每层 MoT 中 VLM tokens 与 video tokens、action tokens 做 trimodal joint attention
  - VLM 输出再投影回 3072 与 WAN 交互

  ---
  6. 图像 T 型拼接与尺寸

  6.1 训练视频

  mp4 是 320(W)×360(H)，已经是 T 型拼接：
  - 顶部：头/前相机（320×180 区域）
  - 底部：左腕 + 右腕 各 160×180 拼接

  训练时 load_video_frames(video_path, indices, target_size=(384, 320))：
  - resize_with_padding：保持长宽比缩放到 384×320，黑边填充
  - 320×360 → scale=min(384/360, 320/320)=min(1.0667, 1.0)=1.0 → 缩放后 360×320，然后上下各 pad 12 行黑边 → 384×320

  6.2 部署拼接（compose_multiview_image）

  top = images[0]  # head, shape (H, W, 3)
  bottom_h = H // 2
  left_w = W // 2; right_w = W - left_w
  left_resized = images[1].resize((left_w, bottom_h))   # 左腕
  right_resized = images[2].resize((right_w, bottom_h)) # 右腕
  bottom_row = concatenate([left_resized, right_resized], axis=1)
  composed = concatenate([top, bottom_row], axis=0)     # T 型
  然后 resize_image_with_padding(composed, (384, 320)) → 同训练一致。

  部署时三路相机顺序固定：images[0]=head, images[1]=left_wrist, images[2]=right_wrist。server_vlm_mask.py 的 get_input_image 支持 images 列表（多图）或单图（已拼接）。

  ---
  7. Flow-Matching 训练与推理

  7.1 训练（training_step）

  视频分支：
  - full_video = cat([first_frame_norm, video_normalized], dim=2) → [B, C, 9, H, W]（1 条件帧 + 8 视频帧）
  - VAE encode → clean_full_latent [B, 48, 3, H', W']（9 帧压缩到 3 latent frames）
  - noisy_video_latent = clean * (1-sigma) + noise * sigma，首帧 teacher-forcing 替换为 clean condition latent
  - target = noise - clean

  action 分支：
  - noisy_actions = actions * (1-sigma_action) + action_noise * sigma_action
  - target = action_noise - actions
  - action 和 video 用独立 timestep（timestep_id vs timestep_id_action）

  7.2 推理（inference_step）

  - num_inference_steps：配置默认 10，服务端 default_steps=10
  - timesteps: torch.linspace(1.0, 0.0, num_inference_steps + 1)，从纯噪声去到 clean
  - 每步：video_t_scaled = t * 1000，action_t_scaled = t * 1000（video 和 action 用同一 t）
  - 视频首帧 latent 始终替换为 condition_frame_latent（teacher forcing）
  - 每步都重新抽取 VLM per-layer features（extract_per_layer_features 调用 num_inference_steps 次，比较耗）

  ---
  8. State 与 Action 的具体处理

  8.1 state

  - initial_state = qpos[condition_frame_idx, qpos_indices] → [20]
  - 训练时 state = batch['initial_state'] → [B, 20]
  - 模型里 state_tokens = state.unsqueeze(1) → [B, 1, state_dim]，与 action tokens 一起进 action_expert.input_encoder

  8.2 action

  - action_sequence = stack([qpos[idx, qpos_indices] for idx in action_indices]) → [16, 20]
  - 训练 target 就是这个 [16, 20] 的绝对关节角（弧度）
  - 推理输出 predicted_actions → [16, 20]，直接是弧度，不需反归一化

  8.3 finetune 权重加载

  finetune.checkpoint_path: /home/ma-user/work/wx1513998/checkpoints/88/mp_rank_00_model_states.pt（17GB DeepSpeed checkpoint）。load_pretrained 时会跳过 action_expert.input_encoder/decoder 形状不匹配的 key（因为从 16 维预训练迁到 20
  维）。

  ---
  9. 部署 Checklist（给真机服务端）

  部署时按这个顺序对齐：

  1. 图像：三路 RGB 相机（head, left_wrist, right_wrist），按 T 型拼接 → resize_with_padding 到 (384, 320) → [0,1] CHW tensor。全链路 RGB，不要 BGR。
  2. state：当前帧 qpos 按 qpos_indices=[24..30, 38, 31..37, 39, 6..9] 取 20 维，顺序 [L_arm(7), L_grip(1), R_arm(7), R_grip(1), torso(4)]。
  3. T5 embedding：
    - 方案 A（推荐先试）：预编码。instruction 文本（含 "The whole scene is in a realistic, industrial art style with three views: ..." prefix）用 T5EncoderModel(text_len=512) 编码成 [seq, 4096] 存 .pt，服务端用 t5_embeddings_path 加载。
    - 方案 B（实时）：服务端加载 T5EncoderModel，每次请求 encoder([instruction], device) → [0] → language_embeddings=[emb]。
  4. VLM 输入：T 型拼接图 + 完整 instruction 文本，用 Qwen3-VL-2B-Instruct 的 AutoProcessor + apply_chat_template(add_generation_prompt=True) + process_vision_info。
  5. 推理：num_inference_steps=10，first_frame=[1,3,384,320] [0,1]，state=[1,20]，language_embeddings=[emb]，vlm_inputs=[dict]。
  6. 输出：predicted_actions [16, 20] 弧度，直接下发，不需反归一化。action chunk 16 步对应 1.6s（10fps）。
  7. action 顺序映射回机器人：输出 [0:7]→left_arm, [7]→L_gripper, [8:15]→right_arm, [15]→R_gripper, [16:20]→torso。
  8. 注意：server_vlm_mask.py 里 DEFAULT_SCENE_PREFIX 是 dobot 的，星海图部署要改成 meta 文件里的 prefix，或直接传完整 instruction。

  ---
  10. 几个容易踩的坑

  1. action 不归一化：如果你误把 stat.json 里的 xinghaitu min/max 用上做 [0,1] 归一化，模型输出会错。训练时是裸弧度。
  2. T5 instruction prefix 必须一致：训练用的是 meta 文件里的完整文本（含 "The whole scene is in a realistic, industrial art style with three views: ..."），部署时 T5 编码的文本必须也是这个完整字符串，不能只传 task 短句。
  3. VLM 图像和 first_frame 是同一张：都是 T 型拼接后的图。VLM 看的是 first_frame（condition frame），不是视频序列。
  4. T 型拼接布局：top=head 全宽，bottom=left_wrist(左半) + right_wrist(右半)。顺序不能换。
  5. resize_with_padding 是黑边填充：不是简单 resize，是保持长宽比 + 中心 padding。部署必须用同一个函数（data/utils/image_utils.py::resize_with_padding）。
  6. VLM add_generation_prompt=True：训练和推理都要加，否则 Qwen3-VL 的 token 序列不一致。
  7. T5 padding 到 512：模型内部 preprocess_t5_embeddings 会 zero-pad 到 512，所以实时编码返回的 [seq, 4096] 不用自己 pad。
  8. qpos_indices 顺序不是 16 维预训练顺序：预训练是 [L_arm(6)+pad+L_grip+R_arm(6)+pad+R_grip] 16 维，星海图是 20 维 [L_arm(7)+L_grip+R_arm(7)+R_grip+torso(4)]，前 16 维布局也不同（多了第 7 维关节），所以 finetune 时 input_encoder/decoder
  形状不匹配会跳过。部署用 20 维。
  9. finetune.checkpoint_path 指向 checkpoints/88/：你说权重已转走，部署时 ckpt_dir 要指向新的 checkpoint 目录（含 mp_rank_00_model_states.pt 或转换后的 HF 格式）。
  10. 视频帧率：训练数据 10fps，global_downsample_rate=1 保持原帧率，action chunk 16 步 = 1.6s。部署执行频率建议 10Hz，replan_steps 可设 16（执行完整个 chunk 再 replan）或更小。
