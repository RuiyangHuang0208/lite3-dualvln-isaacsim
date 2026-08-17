#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：DualVLN / Isaac Sim（开始） ====================
# 本脚本只启动 Lite3、办公室场景、双相机和 ROS 2 桥，不加载 AwareVLN。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate isaaclab232
cd "$REPO_ROOT"

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-Lite3-v0 \
  --num_envs 1 \
  --checkpoint checkpoints/lite3/model_10200.pt \
  --office_scene --ros2_cmd_vel --ros2_camera --real-time \
  "$@"
# ==================== 中文启动区：DualVLN / Isaac Sim（结束） ====================
