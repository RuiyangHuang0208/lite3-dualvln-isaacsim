# DualVLN 与 AwareVLN 启动总览

两套模型已经完全分开，不共用 Python 环境、模型进程或输出协议。

| 项目 | DualVLN | AwareVLN |
|---|---|---|
| Conda 环境 | `dualvln` | `awarevln` |
| 默认代码目录 | `~/lite3_isaac_ws/rl_training` + `~/InternNav` | `~/lite3_isaac_ws/AwareVLN` |
| 输入 | 连续 RGB、下视 RGB、指令、占位 depth/pose/intrinsic | 历史 RGB 队列、指令、可选 previous reasoning |
| 模型帧数 | 历史长度8 | 自动采样/补齐为8帧 |
| 输出 | System 2 action/pixel goal、System 1 trajectory | `<BEGIN_OF_REASONING>` 或参数化文本动作 |
| 机器人控制 | 已接 ROS 2 `/cmd_vel` | 尚未接机器人，先做 inference-only 验证 |

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

```bash
AWAREVLN_INSTRUCTION="go to the sofa" \
  bash scripts/launch/awarevln/01_run_inference.sh
```

AwareVLN 当前只输出 JSON，不会发布 `/cmd_vel`，因此不会与 DualVLN 抢机器人控制权。

## 可覆盖的路径

```bash
export CONDA_ROOT="$HOME/miniconda3"
export ISAAC_PATH="$HOME/isaacsim"
export INTERNNAV_ROOT="$HOME/InternNav"
export AWAREVLN_ROOT="$HOME/lite3_isaac_ws/AwareVLN"
```

不要同时运行两套大模型。切换前先在模型终端按 `Ctrl+C`，再用 `nvidia-smi` 确认显存已释放。
