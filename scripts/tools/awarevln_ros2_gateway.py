#!/usr/bin/env python3
"""AwareVLN ROS 2 网关：Isaac RGB → Python 3.10 推理 → 定时发布 /cmd_vel。"""

import argparse
import json
import os
import socket
import struct
import sys
import time


def receive_exact(connection, size):
    chunks = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise ConnectionError("AwareVLN 推理服务已断开")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def send_frame(connection, metadata, image):
    metadata["image_bytes"] = len(image)
    header = json.dumps(metadata).encode("utf-8")
    connection.sendall(struct.pack("!I", len(header)) + header + image)
    response_size = struct.unpack("!I", receive_exact(connection, 4))[0]
    return json.loads(receive_exact(connection, response_size).decode("utf-8"))


def configure_isaac_ros2():
    """加载 Isaac Sim 自带、与 Python 3.11 匹配的 ROS 2 Jazzy。"""
    isaac_path = os.environ.get("ISAAC_PATH", os.path.join(os.path.expanduser("~"), "isaacsim"))
    ros2_root = os.path.join(isaac_path, "exts", "isaacsim.ros2.bridge", "jazzy")
    python_path = os.path.join(ros2_root, "rclpy")
    library_path = os.path.join(ros2_root, "lib")
    if not os.path.isdir(python_path):
        raise RuntimeError(f"找不到 Isaac Sim ROS 2 Jazzy：{python_path}")
    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ["PYTHONPATH"] = python_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = library_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if os.environ.get("AWAREVLN_GATEWAY_ROS2_READY") != "1":
        os.environ["AWAREVLN_GATEWAY_ROS2_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    sys.path.insert(0, python_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--command-rate", type=float, default=10.0)
    args = parser.parse_args()
    if args.frame_stride < 1 or args.command_rate <= 0:
        parser.error("frame-stride 和 command-rate 必须大于零")

    configure_isaac_ros2()
    import numpy as np
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("awarevln_ros2_gateway")
    cmd_publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    result_publisher = node.create_publisher(String, "/awarevln/result", 10)
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    state = {"image": None, "intrinsic": None, "sequence": 0, "processed": -1}

    # ==================== 中文修改：AwareVLN 只订阅前视 RGB（开始） ====================
    def image_callback(message):
        if message.encoding != "rgb8":
            node.get_logger().error(f"不支持的图像编码：{message.encoding}")
            return
        state["image"] = bytes(message.data)
        state["height"] = message.height
        state["width"] = message.width
        state["sequence"] += 1

    def info_callback(message):
        state["intrinsic"] = np.asarray(message.k, dtype=np.float32).tolist()

    node.create_subscription(Image, "/lite3/camera/rgb", image_callback, qos)
    node.create_subscription(CameraInfo, "/lite3/camera/camera_info", info_callback, qos)
    # ==================== 中文修改：AwareVLN 只订阅前视 RGB（结束） ====================

    stop = Twist()

    def publish_for_duration(command_data):
        """按模型给出的距离/角度执行速度，并在结束后明确发送停止。"""
        command = Twist()
        command.linear.x = float(command_data.get("vx", 0.0))
        command.angular.z = float(command_data.get("wz", 0.0))
        duration = max(0.0, float(command_data.get("duration", 0.0)))
        deadline = time.monotonic() + duration
        period = 1.0 / args.command_rate
        while rclpy.ok() and time.monotonic() < deadline:
            cmd_publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=min(period, 0.05))
            time.sleep(max(0.0, period - 0.05))
        for _ in range(3):
            cmd_publisher.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.03)

    connection = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            sequence = state["sequence"]
            if state["image"] is None or state["intrinsic"] is None:
                continue
            if sequence == state["processed"] or sequence % args.frame_stride:
                continue
            if connection is None:
                try:
                    connection = socket.create_connection((args.host, args.port), timeout=2.0)
                    connection.settimeout(300.0)
                    print(f"[Aware网关] 已连接推理服务：{args.host}:{args.port}")
                except OSError:
                    time.sleep(0.5)
                    continue
            state["processed"] = sequence
            metadata = {
                "sequence": sequence,
                "height": state["height"],
                "width": state["width"],
                "intrinsic": state["intrinsic"],
            }
            try:
                result = send_frame(connection, metadata, state["image"])
                result_message = String()
                result_message.data = json.dumps(result, ensure_ascii=False)
                result_publisher.publish(result_message)
                command = result["cmd_vel"]
                print(
                    f"[Aware网关] frame={sequence} mode={result['mode']} "
                    f"action={result.get('action')} vx={command['vx']:.3f} "
                    f"wz={command['wz']:.3f} duration={command['duration']:.2f}s"
                )
                # reasoning 输出对应零持续时间，机器人保持停止并等待下一帧继续决策。
                publish_for_duration(command)
            except (ConnectionError, BrokenPipeError, OSError) as error:
                print(f"[Aware网关] 推理服务断开：{error}")
                if connection is not None:
                    connection.close()
                connection = None
                publish_for_duration({})
    except KeyboardInterrupt:
        pass
    finally:
        if connection is not None:
            connection.close()
        publish_for_duration({})
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
