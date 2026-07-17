  cd /home/ma-user/work/wx1513998/RoboTwin_eval

  # 全部46个prompt测试（每个5个episode）
  GPU_ID=0 EPISODES_PER_PROMPT=5 \
      bash policy/MotusWanVlmDirectMask_dim16/test_custom_prompts.sh

  # 只测试训练格式的20个prompt（快速）
  GPU_ID=0 python script/test_custom_prompts.py \
      --config policy/MotusWanVlmDirectMask_dim16/deploy_policy.yml \
      --task_name blocks_ranking_custom \
      --task_config demo_randomized \
      --ckpt_setting "/cache/wx1513998/motus/checkpoints_wan_vlm_mask_dim16_0519_8w_pretrain_Robotwin/robotwin_wan_vlm_mask_stage2_dim16/motus_wan_vlm_robotwin_dim16_bs8_lr5e-05/best_action_l2/pytorch_model" \
      --policy_name MotusWanVlmDirectMask_dim16 \
      --wan_path /home/ma-user/work/wx1513998/pretrained_models/Wan2.2-TI2V-5B \
      --vlm_path /home/ma-user/work/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct \
      --seed 42 \
      --episodes_per_prompt 5 \
      --prompt_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19

  # 只测试NOVEL变体（测试泛化能力）
  GPU_ID=0 python script/test_custom_prompts.py \
      --config policy/MotusWanVlmDirectMask_dim16/deploy_policy.yml \
      --task_name blocks_ranking_custom \
      --task_config demo_randomized \
      --ckpt_setting "/cache/wx1513998/motus/checkpoints_wan_vlm_mask_dim16_0519_8w_pretrain_Robotwin/robotwin_wan_vlm_mask_stage2_dim16/motus_wan_vlm_robotwin_dim16_bs8_lr5e-05/best_action_l2/pytorch_model" \
      --policy_name MotusWanVlmDirectMask_dim16 \
      --wan_path /home/ma-user/work/wx1513998/pretrained_models/Wan2.2-TI2V-5B \
      --vlm_path /home/ma-user/work/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct \
      --seed 42 \
      --episodes_per_prompt 3 \
      --prompt_indices 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45

  # 快速验证（只测3个prompt，每个2个episode）
  GPU_ID=0 EPISODES_PER_PROMPT=2 \
      bash policy/MotusWanVlmDirectMask_dim16/test_custom_prompts.sh
