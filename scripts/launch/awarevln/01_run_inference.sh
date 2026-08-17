#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：AwareVLN inference-only（开始） ====================
# 此脚本只读取历史RGB并输出 reasoning/action JSON，不连接 ROS 2，也不控制机器人。
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
AWAREVLN_ROOT="${AWAREVLN_ROOT:-$HOME/lite3_isaac_ws/AwareVLN}"
INSTRUCTION="${AWAREVLN_INSTRUCTION:-go to the sofa}"
IMAGE_DIR="${AWAREVLN_IMAGE_DIR:-$AWAREVLN_ROOT/demo_data/realworld_8frames}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate awarevln
cd "$AWAREVLN_ROOT"

python scripts/inference_only_demo.py \
  --model-path ck/awarevln \
  --image-dir "$IMAGE_DIR" \
  --instruction "$INSTRUCTION" \
  --load-4bit \
  --json-output output/inference_result.json \
  "$@"
# ==================== 中文启动区：AwareVLN inference-only（结束） ====================
