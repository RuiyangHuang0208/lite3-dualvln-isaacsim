#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：DualVLN ROS 2 网关（开始） ====================
# 仅本网关把 DualVLN 的结果发布到 /cmd_vel。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate isaaclab232
cd "$REPO_ROOT"
python scripts/tools/dualvln_ros2_gateway.py "$@"
# ==================== 中文启动区：DualVLN ROS 2 网关（结束） ====================
