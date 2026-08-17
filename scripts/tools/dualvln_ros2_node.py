#!/usr/bin/env python3
"""DualVLN 推理服务（Python 3.9）。

本文件为 DualVLN 仿真接入新增。由于 Isaac ROS 2 使用 Python 3.11，而
DualVLN 官方环境使用 Python 3.9，本进程通过本机 TCP 与 ROS 2 网关通信。
"""

import argparse
import json
import math
import os
import socket
import struct
import sys
import time
from pathlib import Path


def receive_exact(connection, size):
    """从本机连接精确读取指定字节数。"""
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("ROS 2 网关已断开")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_packet(connection):
    metadata_size = struct.unpack("!I", receive_exact(connection, 4))[0]
    metadata = json.loads(receive_exact(connection, metadata_size).decode("utf-8"))
    image = receive_exact(connection, int(metadata["image_bytes"]))
    look_down_image = receive_exact(connection, int(metadata["look_down_bytes"]))
    return metadata, image, look_down_image


def send_packet(connection, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    connection.sendall(struct.pack("!I", len(data)) + data)


def trajectory_to_command(trajectory, max_vx, max_wz):
    """把机器人坐标系 XY 局部轨迹转换为保守的纯追踪速度命令。"""
    if not trajectory:
        return 0.0, 0.0
    x, y = float(trajectory[-1][0]), float(trajectory[-1][1])
    distance = math.hypot(x, y)
    heading = math.atan2(y, max(x, 1.0e-4))
    vx = min(max_vx, 0.6 * distance) * max(0.0, math.cos(heading))
    wz = max(-max_wz, min(max_wz, 1.2 * heading))
    return vx, wz


def action_to_command(actions, max_vx, max_wz):
    """把 System 2 离散动作转换为短时速度命令。"""
    if not actions:
        return 0.0, 0.0
    return {
        0: (0.0, 0.0),
        1: (min(0.20, max_vx), 0.0),
        2: (0.0, max_wz),
        3: (0.0, -max_wz),
        # 安全策略：DualVLN 经常用“↓”表示退让/重新观察。室内闭环暂不允许自动倒车，
        # 避免连续输出该动作时撞到机器人身后的墙壁或家具。
        5: (0.0, 0.0),
    }.get(int(actions[0]), (0.0, 0.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    # ==================== 中文修改：可移植路径参数（开始） ====================
    default_internnav_root = Path(os.environ.get("INTERNNAV_ROOT", Path.home() / "InternNav"))
    parser.add_argument("--internnav-root", type=Path, default=default_internnav_root)
    parser.add_argument("--model-path", type=Path, default=None)
    # ==================== 中文修改：可移植路径参数（结束） ====================
    parser.add_argument("--max-vx", type=float, default=0.30)
    parser.add_argument("--max-wz", type=float, default=0.40)
    args = parser.parse_args()
    if args.max_vx <= 0.0 or args.max_wz <= 0.0:
        parser.error("速度限制必须大于零")
    return args


def main():
    args = parse_args()
    import numpy as np

    # ==================== 中文修改：可移植路径解析（开始） ====================
    internnav_root = args.internnav_root.expanduser().resolve()
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else internnav_root / "checkpoints/InternVLA-N1-DualVLN"
    )
    if not (internnav_root / "internnav").is_dir():
        raise FileNotFoundError(f"找不到 InternNav：{internnav_root}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到 DualVLN 权重：{model_path}")
    # ==================== 中文修改：可移植路径解析（结束） ====================
    sys.path.insert(0, str(internnav_root))
    sys.path.insert(0, str(internnav_root / "src/diffusion-policy"))
    os.chdir(internnav_root / "scripts/notebooks")
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    # ==================== DualVLN 模型参数：保持与官方 demo 一致（开始） ====================
    class AgentArgs:
        device = "cuda:0"
        model_path = str(model_path)
        resize_w = 384
        resize_h = 384
        num_history = 8
        plan_step_gap = 4
    # ==================== DualVLN 模型参数：保持与官方 demo 一致（结束） ====================

    print(f"[DualVLN] 正在加载模型：{model_path}")
    agent = InternVLAN1AsyncAgent(AgentArgs())
    agent.reset()
    print("[DualVLN] 模型已就绪")

    # ==================== Python 3.9 推理服务：本机 TCP（开始） ====================
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[DualVLN] 等待 ROS 2 网关：{args.host}:{args.port}")
        while True:
            connection, address = server.accept()
            print(f"[DualVLN] ROS 2 网关已连接：{address}")
            try:
                with connection:
                    while True:
                        metadata, image_bytes, look_down_bytes = receive_packet(connection)
                        height, width = int(metadata["height"]), int(metadata["width"])
                        rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
                        look_down_rgb = np.frombuffer(look_down_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
                        intrinsic = np.eye(4, dtype=np.float32)
                        intrinsic[:3, :3] = np.asarray(metadata["intrinsic"], dtype=np.float32).reshape(3, 3)
                        depth = np.zeros((height, width), dtype=np.float32)
                        pose = np.eye(4, dtype=np.float32)
                        started = time.monotonic()

                        # ==================== DualVLN 推理及 trajectory→cmd_vel（开始） ====================
                        output = agent.step(rgb, depth, pose, args.instruction, intrinsic=intrinsic)
                        # System 2 的“↓”(动作 5)表示请求向下观察，而不是让机器人倒车。
                        # 立即送入独立下视相机图像，使模型继续生成 pixel goal / trajectory。
                        if output.output_action and 5 in output.output_action:
                            output = agent.step(
                                look_down_rgb,
                                depth,
                                pose,
                                args.instruction,
                                intrinsic=intrinsic,
                                look_down=True,
                            )
                        trajectory = None
                        actions = None
                        if output.output_trajectory is not None:
                            trajectory = output.output_trajectory.detach().cpu().numpy().tolist()
                            vx, wz = trajectory_to_command(trajectory, args.max_vx, args.max_wz)
                        else:
                            actions = output.output_action
                            vx, wz = action_to_command(actions, args.max_vx, args.max_wz)
                        result = {
                            "sequence": metadata["sequence"],
                            "llm_output": agent.llm_output,
                            "pixel_goal": output.output_pixel,
                            "trajectory": trajectory,
                            "actions": actions,
                            "cmd_vel": {"vx": vx, "wz": wz},
                            "inference_seconds": time.monotonic() - started,
                        }
                        send_packet(connection, result)
                        print(
                            f"[DualVLN] frame={metadata['sequence']} llm={agent.llm_output!r} "
                            f"vx={vx:.3f} wz={wz:.3f} dt={result['inference_seconds']:.2f}s"
                        )
                        # ==================== DualVLN 推理及 trajectory→cmd_vel（结束） ====================
            except (ConnectionError, BrokenPipeError, OSError) as error:
                print(f"[DualVLN] 网关断开，等待重连：{error}")
    # ==================== Python 3.9 推理服务：本机 TCP（结束） ====================


if __name__ == "__main__":
    main()
