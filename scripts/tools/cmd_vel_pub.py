#!/usr/bin/env python3
"""Publish geometry_msgs/Twist commands to the Lite3 /cmd_vel bridge."""

import argparse
import os
import sys
import time


def configure_isaac_ros2():
    """Load the ROS 2 Jazzy runtime bundled with Isaac Sim standalone."""
    # 中文修改：避免写死开发机用户名，可通过 ISAAC_PATH 指定安装位置。
    isaac_path = os.environ.get("ISAAC_PATH", os.path.join(os.path.expanduser("~"), "isaacsim"))
    ros2_root = os.path.join(isaac_path, "exts", "isaacsim.ros2.bridge", "jazzy")
    python_path = os.path.join(ros2_root, "rclpy")
    library_path = os.path.join(ros2_root, "lib")
    if not os.path.isdir(python_path):
        raise RuntimeError(f"Isaac Sim internal ROS 2 Jazzy was not found at: {python_path}")

    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ["PYTHONPATH"] = python_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = library_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if os.environ.get("LITE3_ROS2_PUBLISHER_READY") != "1":
        os.environ["LITE3_ROS2_PUBLISHER_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    sys.path.insert(0, python_path)


def main():
    parser = argparse.ArgumentParser(description="Publish velocity commands for Lite3 in Isaac Sim.")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity in m/s.")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw velocity in rad/s.")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate in Hz.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Publish duration in seconds; 0 means until Ctrl+C.",
    )
    args = parser.parse_args()
    if args.rate <= 0.0:
        parser.error("--rate must be greater than zero.")
    if args.duration < 0.0:
        parser.error("--duration cannot be negative.")

    configure_isaac_ros2()

    import rclpy
    from geometry_msgs.msg import Twist

    rclpy.init()
    node = rclpy.create_node("lite3_cmd_vel_publisher")
    publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    command = Twist()
    command.linear.x = args.vx
    command.linear.y = args.vy
    command.angular.z = args.wz
    period = 1.0 / args.rate
    started_at = time.monotonic()

    print(
        f"Publishing /cmd_vel: vx={args.vx:.3f}, vy={args.vy:.3f}, "
        f"wz={args.wz:.3f}, rate={args.rate:.1f} Hz"
    )
    try:
        while rclpy.ok():
            publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=min(period, 0.05))
            if args.duration > 0.0 and time.monotonic() - started_at >= args.duration:
                break
            time.sleep(max(0.0, period - 0.05))
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        for _ in range(5):
            publisher.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.05)
        print("Published stop command.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
