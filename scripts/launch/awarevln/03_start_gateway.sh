#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：AwareVLN ROS 2 网关（开始） ====================
# 只在没有运行 DualVLN 网关时启动，否则两者会同时发布 /cmd_vel。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate isaaclab232
cd "$REPO_ROOT"
python scripts/tools/awarevln_ros2_gateway.py --port "${AWAREVLN_PORT:-8766}" "$@"
# ==================== 中文启动区：AwareVLN ROS 2 网关（结束） ====================
