#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/train_smolvla.sh HF_USER/dataset data/lerobot_dataset outputs/smolvla" >&2
  exit 2
fi
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features='{}' \
  --dataset.repo_id="$1" \
  --dataset.root="$2" \
  --output_dir="$3" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=5000 \
  --save_freq=1000 \
  --job_name=phone_to_panda \
  --wandb.enable=false
