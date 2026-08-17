# Lite3 + Isaac Sim + ROS 2 + DualVLN 仿真使用说明

本仓库保存已经接通的仿真闭环：Isaac Sim 办公室前视/下视 RGB → ROS 2 网关 → Python 3.9 DualVLN → `/cmd_vel` → Lite3 locomotion policy。

## 1. 克隆

```bash
git clone --recurse-submodules https://github.com/RuiyangHuang0208/lite3-dualvln-isaacsim.git
cd lite3-dualvln-isaacsim
```

需要 Isaac Sim 5.1、Isaac Lab 2.3.2，以及能够运行本仓库 Lite3 环境的 Python 3.11 环境。仓库已包含本次验证使用的 `checkpoints/lite3/model_10200.pt`。

## 2. 安装 InternNav 与权重

```bash
cd ~
git clone --recursive https://github.com/InternRobotics/InternNav.git
cd InternNav
git submodule update --init --recursive

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n dualvln python=3.9 -y
conda activate dualvln
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.0 diffusers==0.31.0 accelerate==1.10.1 \
  opencv-python==4.10.0.82 pillow==10.4.0 numpy==1.26.4 gym==0.23.1 \
  imageio==2.37.0 imageio-ffmpeg==0.6.0 ftfy==6.3.1 scipy matplotlib
pip install -e .

mkdir -p checkpoints
pip install -U huggingface_hub
hf download InternRobotics/InternVLA-N1-DualVLN \
  --local-dir checkpoints/InternVLA-N1-DualVLN
```

应用 RTX 5090/联合运行显存补丁（在本仓库目录执行）：

```bash
git -C "$HOME/InternNav" apply \
  "$(pwd)/patches/internnav_dualvln_isaac_memory.patch"
```

如果仓库不在 `~/InternNav`，启动前设置：

```bash
export INTERNNAV_ROOT=/你的路径/InternNav
```

## 3. 启动（3 个终端）

先设置 Isaac Sim 路径；默认是 `~/isaacsim`：

```bash
export ISAAC_PATH="$HOME/isaacsim"
```

终端 1：启动 Lite3、办公室、双相机和 ROS 2 控制。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab232
cd ~/lite3-dualvln-isaacsim
export ISAAC_PATH="$HOME/isaacsim"
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-Lite3-v0 \
  --num_envs 1 \
  --checkpoint checkpoints/lite3/model_10200.pt \
  --office_scene --ros2_cmd_vel --ros2_camera --real-time
```

终端 2：启动 Python 3.9 DualVLN 推理服务。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dualvln
cd ~/lite3-dualvln-isaacsim
export INTERNNAV_ROOT="$HOME/InternNav"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/tools/dualvln_ros2_node.py \
  --instruction "go to the sofa" \
  --max-vx 0.15 --max-wz 0.20
```

终端 3：启动 ROS 2/TCP 网关。它使用 Isaac Sim 自带的 ROS 2 Jazzy 库。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab232
cd ~/lite3-dualvln-isaacsim
export ISAAC_PATH="$HOME/isaacsim"
python scripts/tools/dualvln_ros2_gateway.py
```

点击 Isaac Sim 的 Play。正常情况下三个终端会分别显示 `[ROS2 CMD]`、`[DualVLN]` 和 `[网关]`。

## 4. 键盘/ROS 2 独立测试

键盘测试可在终端 1 增加 `--keyboard`。直接测试 `/cmd_vel`：

```bash
python scripts/tools/cmd_vel_pub.py --vx 0.15 --duration 5
```

自动导航测试时不要同时使用键盘，避免两路速度命令相加。停止推理或网关后，安全超时会将速度归零。

## 5. 未包含的内容

- DualVLN 权重约 16.8 GB，请按上面的 Hugging Face 命令下载。
- NVIDIA 办公室 USD 在首次运行时由 Isaac Sim 在线加载并缓存。
- Isaac Sim、Isaac Lab 和 Conda 环境本身不纳入 Git。
