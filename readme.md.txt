启动 server：

  CUDA_VISIBLE_DEVICES=4 python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/server_vlm_mask.py \
    --model_config /home/ma-user/work/wwx1484778/Our/Motus/configs/robotwin_wan_vlm_mask_dobot_c.yaml \
    --ckpt_dir /cache/wwx1484778/motus/checkpoints_wan_vlm_mask_dobot_c_0513/robotwin_wan_vlm_mask_dobot_c/motus_wan_vlm_dobot_bs8_lr5e-05/checkpoint_step_20000/pytorch_model \
    --wan_path /cache/wwx1484778/motus_weights/Wan2.2-TI2V-5B \
    --vlm_path /cache/wwx1484778/motus_weights/Qwen3-VL-2B-Instruct \
    --dataset_name dobot_cook_vegetable \
    --port 6789

  启动 client 测试：

  # 健康检查
  python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/client.py \
    --url http://localhost:6789 \
    --test connectivity

  # 推理测试
  python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/client.py \
    --url http://localhost:6789 \
    --test inference \
    --instruction "The whole scene is in a realistic, indoor environment with three camera views: a fixed top camera, a movable left arm wrist camera, and a movable right arm wrist camera. The Dobot dual-arm robot is currently performing the following task: cook vegetable" \
    --t5_embeddings_path /cache/wwx1484778/Dobot/dobot_cook_vegetable_full/dobot_cook_vegetable_full/episode_000000/umt5_wan/trajectory.pt \
    --image /cache/wwx1484778/Dobot/dobot_first_frame.jpg
 需要输入state

