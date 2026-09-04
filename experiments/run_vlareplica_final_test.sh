#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1

exec torchrun --standalone --nproc-per-node=2 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-SO100_101 \
  vlareplica_train \
  --validation_mixture=vlareplica_val \
  --max_duration=5 \
  --device_batch_size=1 \
  --global_batch_size=2 \
  --num_workers=0 \
  --pin_memory=false \
  --save_folder=/metadisk/may/molmoact2-checkpoints/final-test-5-v2 \
  --save_num_checkpoints_to_keep=0 \
  --eval_interval=5 \
  --max_loss_examples=8 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --action_expert_learning_rate=5e-5
