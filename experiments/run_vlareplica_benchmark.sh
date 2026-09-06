#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

gpu_ids=${GPU_IDS:-0,1,2,3}
nproc_per_node=${NPROC_PER_NODE:-4}
global_batch_size=${GLOBAL_BATCH_SIZE:-16}
device_batch_size=${DEVICE_BATCH_SIZE:-2}
storage_root=${VLA_STORAGE:-/data3/may}
run_dir=${RUN_DIR:-$storage_root/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16}
run_log=${RUN_LOG:-$storage_root/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16.log}

if [[ -e "$run_dir/config.yaml" ]]; then
  echo "Run folder already exists: $run_dir" >&2
  echo "Set RUN_DIR and RUN_LOG to new paths, or intentionally resume with the raw command in cmds.md." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=${WANDB_PROJECT:-molmoact2-vlareplica}
export WANDB_ENTITY=${WANDB_ENTITY:-local}
export WANDB_MODE=${WANDB_MODE:-offline}

echo "GPUs: $gpu_ids ($nproc_per_node processes)"
echo "Effective global batch: $global_batch_size"
echo "Per-GPU device batch: $device_batch_size"
echo "Run directory: $run_dir"
echo "Console log: $run_log"

torchrun --standalone --nproc-per-node="$nproc_per_node" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-SO100_101 \
  vlareplica \
  --frame_loading_backend=av \
  --max_duration=40000 \
  --device_batch_size="$device_batch_size" \
  --global_batch_size="$global_batch_size" \
  --num_workers=4 \
  --pin_memory=true \
  --log_interval=10 \
  --save_folder="$run_dir" \
  --save_interval=5000 \
  --save_num_checkpoints_to_keep=2 \
  --eval_interval=-1 \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --save_final_optim=false \
  --save_final_unsharded_checkpoint=true 2>&1 | tee "$run_log"
