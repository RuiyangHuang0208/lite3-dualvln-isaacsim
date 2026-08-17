#!/usr/bin/env bash
set -euo pipefail

# ==================== 中文启动区：AwareVLN / Isaac Sim（开始） ====================
# AwareVLN 复用同一 Lite3 办公室和前视相机，但不会启动 DualVLN 模型。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
exec bash "$REPO_ROOT/scripts/launch/dualvln/01_start_isaac_sim.sh" "$@"
# ==================== 中文启动区：AwareVLN / Isaac Sim（结束） ====================
