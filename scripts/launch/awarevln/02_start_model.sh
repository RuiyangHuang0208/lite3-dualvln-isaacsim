#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：AwareVLN 实时模型服务（开始） ====================
# 输入历史前视 RGB；输出 reasoning 或 forward/left/right/stop 参数化动作。
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
AWAREVLN_ROOT="${AWAREVLN_ROOT:-$HOME/lite3_isaac_ws/AwareVLN}"
INSTRUCTION="${AWAREVLN_INSTRUCTION:-go to the sofa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate awarevln
cd "$AWAREVLN_ROOT"
python scripts/awarevln_ros2_node.py \
  --instruction "$INSTRUCTION" \
  --port "${AWAREVLN_PORT:-8766}" \
  --max-vx "${AWAREVLN_MAX_VX:-0.15}" \
  --max-wz "${AWAREVLN_MAX_WZ:-0.20}" \
  "$@"
# ==================== 中文启动区：AwareVLN 实时模型服务（结束） ====================
