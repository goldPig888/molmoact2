#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run_dir=/metadisk/may/molmoact2-checkpoints/pilot-200-live
run_log=/metadisk/may/molmoact2-checkpoints/pilot-200-live.log

if [[ -e "$run_dir/config.yaml" ]]; then
  echo "Run folder already exists: $run_dir" >&2
  echo "Choose a new folder or remove the old run before restarting." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1

echo "Writing the live console log to $run_log"
echo "Open a second terminal and run:"
echo "python scripts/monitor_vlareplica_training.py"

torchrun --standalone --nproc-per-node=2 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-SO100_101 \
  vlareplica_train \
  --validation_mixture=vlareplica_val \
  --max_duration=200 \
  --device_batch_size=1 \
  --global_batch_size=2 \
  --num_workers=0 \
  --pin_memory=false \
  --log_interval=1 \
  --save_folder="$run_dir" \
  --save_interval=100 \
  --save_num_checkpoints_to_keep=3 \
  --eval_interval=100 \
  --max_loss_examples=96 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --action_expert_learning_rate=5e-5 2>&1 | tee "$run_log"
