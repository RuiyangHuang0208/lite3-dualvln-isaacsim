#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：DualVLN 模型服务（开始） ====================
# 输入：主视RGB + 下视RGB + 指令；输出：action/pixel goal/trajectory。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/InternNav}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
INSTRUCTION="${DUALVLN_INSTRUCTION:-go to the sofa}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dualvln
cd "$REPO_ROOT"

python scripts/tools/dualvln_ros2_node.py \
  --instruction "$INSTRUCTION" \
  --max-vx "${DUALVLN_MAX_VX:-0.15}" \
  --max-wz "${DUALVLN_MAX_WZ:-0.20}" \
  "$@"
# ==================== 中文启动区：DualVLN 模型服务（结束） ====================
