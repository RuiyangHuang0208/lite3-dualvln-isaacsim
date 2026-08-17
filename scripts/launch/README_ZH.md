# DualVLN 与 AwareVLN 启动总览

两套模型已经完全分开，不共用 Python 环境、模型进程或输出协议。

| 项目 | DualVLN | AwareVLN |
|---|---|---|
| Conda 环境 | `dualvln` | `awarevln` |
| 默认代码目录 | `~/lite3_isaac_ws/rl_training` + `~/InternNav` | `~/lite3_isaac_ws/AwareVLN` |
| 输入 | 连续 RGB、下视 RGB、指令、占位 depth/pose/intrinsic | 历史 RGB 队列、指令、可选 previous reasoning |
| 模型帧数 | 历史长度8 | 自动采样/补齐为8帧 |
| 输出 | System 2 action/pixel goal、System 1 trajectory | `<BEGIN_OF_REASONING>` 或参数化文本动作 |
| 机器人控制 | 已接 ROS 2 `/cmd_vel` | 已接独立 ROS 2 网关，以动作距离/角度定时控制 |

## DualVLN：三个终端

终端1：

```bash
bash scripts/launch/dualvln/01_start_isaac_sim.sh
```

终端2：

```bash
DUALVLN_INSTRUCTION="go to the sofa" \
  bash scripts/launch/dualvln/02_start_model.sh
```

终端3：

```bash
bash scripts/launch/dualvln/03_start_gateway.sh
```

## AwareVLN：独立推理

首次安装官方 AwareVLN 后，在本仓库根目录应用 RTX 5090 和正确输入输出接口补丁：

```bash
git -C "$HOME/lite3_isaac_ws/AwareVLN" am \
  "$(pwd)"/patches/awarevln/*.patch
```

这两个补丁只修改独立的 AwareVLN 仓库，不会修改 DualVLN/InternNav。

```bash
AWAREVLN_INSTRUCTION="go to the sofa" \
  bash scripts/launch/awarevln/00_run_inference_only.sh
```

## AwareVLN：Isaac Sim 闭环（三个终端）

```bash
# 终端1：Lite3、办公室和 ROS 2 前视相机
bash scripts/launch/awarevln/01_start_isaac_sim.sh

# 终端2：AwareVLN Python 3.10 实时推理服务（默认4-bit）
AWAREVLN_INSTRUCTION="go to the sofa" \
  bash scripts/launch/awarevln/02_start_model.sh

# 终端3：AwareVLN 独立网关，发布 /cmd_vel
bash scripts/launch/awarevln/03_start_gateway.sh
```

AwareVLN reasoning 输出时机器人停止；action 输出转换规则为：前进25/50/75cm，左右转15/30/45度，或停止。网关按速度和持续时间执行后主动发送零速度。

**禁止同时运行 `dualvln/03_start_gateway.sh` 和 `awarevln/03_start_gateway.sh`**，否则两个节点会同时发布 `/cmd_vel`。

## 可覆盖的路径

```bash
export CONDA_ROOT="$HOME/miniconda3"
export ISAAC_PATH="$HOME/isaacsim"
export INTERNNAV_ROOT="$HOME/InternNav"
export AWAREVLN_ROOT="$HOME/lite3_isaac_ws/AwareVLN"
```

不要同时运行两套大模型。切换前先在模型终端按 `Ctrl+C`，再用 `nvidia-smi` 确认显存已释放。
