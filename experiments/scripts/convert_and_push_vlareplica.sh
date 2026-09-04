#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

hf_repo_id=${1:-}
run_dir=${RUN_DIR:-/metadisk/may/molmoact2-checkpoints/vlareplica-molmoact2-40k-bs16}
source_checkpoint=${SOURCE_CHECKPOINT:-$run_dir/step40000-unsharded}
hf_output=${HF_OUTPUT:-$run_dir/step40000-hf}
private=${PRIVATE:-true}

if [[ -z "$hf_repo_id" ]]; then
  echo "Usage: $0 <huggingface-user-or-org/model-name>" >&2
  exit 2
fi
if [[ ! -f "$source_checkpoint/config.yaml" ]]; then
  echo "Pushable checkpoint not found: $source_checkpoint" >&2
  exit 2
fi

hf auth whoami

if [[ ! -f "$hf_output/config.json" ]]; then
  python -m olmo.hf_model.convert_molmoact2_to_hf \
    "$source_checkpoint" \
    "$hf_output"
else
  echo "Using existing converted checkpoint: $hf_output"
fi

upload_args=(upload-large-folder "$hf_repo_id" "$hf_output")
if [[ "$private" == "true" ]]; then
  upload_args+=(--private)
fi
hf "${upload_args[@]}"
