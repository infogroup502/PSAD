#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'CUDA is unavailable. Activate the CUDA-enabled PSAD Conda environment first.')"

"${PYTHON_BIN}" main.py \
  --dataset SKAB \
  --data_path SKAB \
  --mode train \
  --temporal_length 20 \
  --step 1 \
  --batch_size 256 \
  --num_epochs 10 \
  --pretrain_epochs 0 \
  --lr 0.0002 \
  --threshold_ratio 0.993 \
  --embed_dim 128 \
  --dropout 0.1 \
  --center_count 80 \
  --candidate_count 3 \
  --conv_channels 96 \
  --cluster_loss_weight 0.1 \
  --top_beta 5 \
  --device cuda \
  --use_gpu true \
  --deterministic true \
  --seed 42
