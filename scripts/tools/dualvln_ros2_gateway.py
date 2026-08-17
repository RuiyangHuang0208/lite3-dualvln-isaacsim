#!/usr/bin/env python3
"""DualVLN ROS 2 网关（Isaac Sim Python 3.11 环境）。

订阅 Isaac RGB/CameraInfo，把最新帧发送给 Python 3.9 推理服务，并将返回结果
发布为 /cmd_vel 和 /dualvln/result。本文件为 DualVLN 仿真接入新增。
"""

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
            raise ConnectionError("DualVLN 推理服务已断开")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def send_frame(connection, metadata, image, look_down_image):
    metadata["image_bytes"] = len(image)
    metadata["look_down_bytes"] = len(look_down_image)
    header = json.dumps(metadata).encode("utf-8")
    connection.sendall(struct.pack("!I", len(header)) + header + image + look_down_image)
    response_size = struct.unpack("!I", receive_exact(connection, 4))[0]
    return json.loads(receive_exact(connection, response_size).decode("utf-8"))


# ==================== Isaac Sim 内置 ROS 2 环境配置（开始） ====================
def configure_isaac_ros2():
    """加载与 Python 3.11 匹配的 Isaac Sim ROS 2 Jazzy 运行库。"""
    # 中文修改：避免写死开发机用户名，可通过 ISAAC_PATH 指定安装位置。
    isaac_path = os.environ.get("ISAAC_PATH", os.path.join(os.path.expanduser("~"), "isaacsim"))
    ros2_root = os.path.join(isaac_path, "exts", "isaacsim.ros2.bridge", "jazzy")
    python_path = os.path.join(ros2_root, "rclpy")
    library_path = os.path.join(ros2_root, "lib")
    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ["PYTHONPATH"] = python_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = library_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if os.environ.get("DUALVLN_GATEWAY_ROS2_READY") != "1":
        os.environ["DUALVLN_GATEWAY_ROS2_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    sys.path.insert(0, python_path)
# ==================== Isaac Sim 内置 ROS 2 环境配置（结束） ====================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    configure_isaac_ros2()

    import numpy as np
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("dualvln_ros2_gateway")
    cmd_publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    result_publisher = node.create_publisher(String, "/dualvln/result", 10)
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    state = {"image": None, "look_down": None, "intrinsic": None, "sequence": 0, "processed": -1}

    # ==================== ROS 2 最新帧缓存（开始） ====================
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

    def look_down_callback(message):
        if message.encoding == "rgb8":
            state["look_down"] = bytes(message.data)

    node.create_subscription(Image, "/lite3/camera/rgb", image_callback, qos)
    node.create_subscription(CameraInfo, "/lite3/camera/camera_info", info_callback, qos)
    node.create_subscription(Image, "/lite3/camera/look_down", look_down_callback, qos)
    # ==================== ROS 2 最新帧缓存（结束） ====================

    stop = Twist()
    connection = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            sequence = state["sequence"]
            if state["image"] is None or state["look_down"] is None or state["intrinsic"] is None:
                continue
            if sequence == state["processed"] or sequence % args.frame_stride:
                continue
            if connection is None:
                try:
                    connection = socket.create_connection((args.host, args.port), timeout=2.0)
                    connection.settimeout(120.0)
                    print(f"[网关] 已连接 DualVLN：{args.host}:{args.port}")
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
                # ==================== 推理结果发布到 ROS 2（开始） ====================
                result = send_frame(connection, metadata, state["image"], state["look_down"])
                command = Twist()
                command.linear.x = float(result["cmd_vel"]["vx"])
                command.angular.z = float(result["cmd_vel"]["wz"])
                cmd_publisher.publish(command)
                result_message = String()
                result_message.data = json.dumps(result, ensure_ascii=False)
                result_publisher.publish(result_message)
                print(
                    f"[网关] frame={sequence} vx={command.linear.x:.3f} "
                    f"wz={command.angular.z:.3f}"
                )
                # ==================== 推理结果发布到 ROS 2（结束） ====================
            except (ConnectionError, BrokenPipeError, OSError) as error:
                print(f"[网关] 推理服务断开：{error}")
                connection.close()
                connection = None
    except KeyboardInterrupt:
        pass
    finally:
        if connection is not None:
            connection.close()
        for _ in range(5):
            cmd_publisher.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
