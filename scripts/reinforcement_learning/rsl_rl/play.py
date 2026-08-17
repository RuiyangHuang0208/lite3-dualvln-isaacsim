# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import threading

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--ros2_cmd_vel",
    action="store_true",
    default=False,
    help="Use ROS 2 /cmd_vel as velocity command.",
)
# ==================== DualVLN 仿真接入：前视相机与安全参数（开始） ====================
parser.add_argument(
    "--ros2_camera",
    action="store_true",
    default=False,
    help="Publish the robot front camera on ROS 2 topics.",
)
parser.add_argument("--camera_rate", type=float, default=5.0, help="Front camera ROS 2 publish rate in Hz.")
parser.add_argument("--cmd_vel_timeout", type=float, default=2.0, help="Stop when /cmd_vel is stale for this many seconds.")
# ==================== DualVLN 仿真接入：前视相机与安全参数（结束） ====================
# ==================== DualVLN 办公室场景：启动参数（开始） ====================
parser.add_argument("--office_scene", action="store_true", help="加载 NVIDIA Isaac 官方办公室场景。")
parser.add_argument("--spawn_x", type=float, default=0.0, help="Lite3 在办公室中的初始 X 坐标。")
parser.add_argument("--spawn_y", type=float, default=0.0, help="Lite3 在办公室中的初始 Y 坐标。")
parser.add_argument("--spawn_yaw", type=float, default=0.0, help="Lite3 初始偏航角，单位为弧度。")
# ==================== DualVLN 办公室场景：启动参数（结束） ====================
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# ==================== DualVLN 仿真接入：加载 Isaac Sim 内置 ROS 2（开始） ====================
if args_cli.ros2_cmd_vel or args_cli.ros2_camera:
    # Isaac Sim standalone ships its own ROS 2 Jazzy Python modules and shared libraries.
    # 中文修改：默认使用当前用户目录，换机器后也可通过 ISAAC_PATH 覆盖。
    isaac_path = os.environ.get("ISAAC_PATH", os.path.join(os.path.expanduser("~"), "isaacsim"))
    ros2_bridge_path = os.path.join(isaac_path, "exts", "isaacsim.ros2.bridge", "jazzy")
    ros2_python_path = os.path.join(ros2_bridge_path, "rclpy")
    ros2_library_path = os.path.join(ros2_bridge_path, "lib")
    if not os.path.isdir(ros2_python_path):
        parser.error(f"Isaac Sim internal ROS 2 Jazzy was not found at: {ros2_python_path}")
    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ["PYTHONPATH"] = ros2_python_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = ros2_library_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    # The ELF loader reads LD_LIBRARY_PATH when the process starts. Re-exec once so
    # rclpy native modules can resolve librcl_action.so and the other ROS 2 libraries.
    if os.environ.get("LITE3_ROS2_ENV_READY") != "1":
        os.environ["LITE3_ROS2_ENV_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    sys.path.insert(0, ros2_python_path)
# ==================== DualVLN 仿真接入：加载 Isaac Sim 内置 ROS 2（结束） ====================
# always enable cameras to record video
if args_cli.video or args_cli.ros2_camera:
    # 启用 RTX 渲染，否则前视 RGB 传感器不会产生图像。
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rl_utils import camera_follow

import gymnasium as gym
import numpy as np
import time
import torch

if args_cli.ros2_cmd_vel or args_cli.ros2_camera:
    import rclpy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import CameraInfo, Image
else:
    rclpy = None
    Twist = object

from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


# ==================== DualVLN 仿真接入：ROS 2 双向桥（开始） ====================
class Lite3Ros2Bridge:
    """Exchange camera images and velocity commands with ROS 2."""

    def __init__(self, command_limits: tuple[float, float, float], subscribe_cmd_vel: bool, publish_camera: bool):
        rclpy.init(args=None)
        self.node = rclpy.create_node("lite3_isaac_bridge")
        self.command = torch.zeros(3, dtype=torch.float32)
        self.command_limits = torch.tensor(command_limits, dtype=torch.float32)
        self.command_timeout = args_cli.cmd_vel_timeout
        # 超过该时间未收到新速度时，自动向策略提供零速度，避免失联后持续行走。
        self.last_command_time = 0.0
        self.lock = threading.Lock()
        self.subscription = None
        if subscribe_cmd_vel:
            self.subscription = self.node.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)
        self.image_publisher = self.node.create_publisher(Image, "/lite3/camera/rgb", 2) if publish_camera else None
        self.look_down_publisher = (
            self.node.create_publisher(Image, "/lite3/camera/look_down", 2) if publish_camera else None
        )
        self.info_publisher = (
            self.node.create_publisher(CameraInfo, "/lite3/camera/camera_info", 2) if publish_camera else None
        )
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        print("[INFO] ROS 2 bridge enabled on node 'lite3_isaac_bridge'.")

    def _spin(self):
        try:
            rclpy.spin(self.node)
        except rclpy.executors.ExternalShutdownException:
            pass

    def cmd_vel_callback(self, msg: Twist):
        command = torch.tensor(
            [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)],
            dtype=torch.float32,
        )
        command = torch.clamp(command, min=-self.command_limits, max=self.command_limits)
        with self.lock:
            changed = not torch.equal(self.command, command)
            self.command.copy_(command)
            self.last_command_time = time.monotonic()
        if changed:
            print(
                f"[ROS2 CMD] vx={command[0].item():.3f} "
                f"vy={command[1].item():.3f} wz={command[2].item():.3f}"
            )

    def advance(self):
        with self.lock:
            if self.last_command_time == 0.0 or time.monotonic() - self.last_command_time > self.command_timeout:
                return torch.zeros_like(self.command)
            return self.command.clone()

    def publish_camera(self, rgb: np.ndarray, intrinsic: np.ndarray, look_down_rgb: np.ndarray | None = None):
        """把 Isaac 相机张量转换成标准 ROS 2 Image 和 CameraInfo。"""
        if self.image_publisher is None:
            return
        stamp = self.node.get_clock().now().to_msg()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        image_msg = Image()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = "lite3_front_camera"
        image_msg.height, image_msg.width = rgb.shape[:2]
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = False
        image_msg.step = image_msg.width * 3
        image_msg.data = rgb.tobytes()
        info_msg = CameraInfo()
        info_msg.header = image_msg.header
        info_msg.height = image_msg.height
        info_msg.width = image_msg.width
        info_msg.k = intrinsic.reshape(-1).astype(float).tolist()
        info_msg.p = [
            float(intrinsic[0, 0]), 0.0, float(intrinsic[0, 2]), 0.0,
            0.0, float(intrinsic[1, 1]), float(intrinsic[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.image_publisher.publish(image_msg)
        self.info_publisher.publish(info_msg)
        if look_down_rgb is not None:
            # DualVLN 输出“↓”时需要额外的向下观察；使用独立 ROS 2 图像话题提供该视角。
            look_down_rgb = np.ascontiguousarray(look_down_rgb[..., :3], dtype=np.uint8)
            look_down_msg = Image()
            look_down_msg.header = image_msg.header
            look_down_msg.header.frame_id = "lite3_look_down_camera"
            look_down_msg.height, look_down_msg.width = look_down_rgb.shape[:2]
            look_down_msg.encoding = "rgb8"
            look_down_msg.is_bigendian = False
            look_down_msg.step = look_down_msg.width * 3
            look_down_msg.data = look_down_rgb.tobytes()
            self.look_down_publisher.publish(look_down_msg)

    def shutdown(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.thread.join(timeout=1.0)
# ==================== DualVLN 仿真接入：ROS 2 双向桥（结束） ====================


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ==================== DualVLN 办公室场景：替换粗糙地形（开始） ====================
    if args_cli.office_scene:
        office_usd = (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
            "Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
        )
        env_cfg.scene.num_envs = 1
        env_cfg.scene.env_spacing = 1.0
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.usd_path = office_usd
        env_cfg.scene.terrain.use_terrain_origins = False
        env_cfg.scene.terrain.env_spacing = 1.0
        # 主 Viewport 使用机器人相对坐标，避免默认世界相机被办公室墙体遮挡。
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-1.5, 0.0, 0.9)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
        # 办公室不是程序化地形，因此关闭依赖 terrain_generator 的课程和越界判定。
        env_cfg.curriculum.terrain_levels = None
        env_cfg.terminations.terrain_out_of_bounds = None
        print(f"[INFO] 加载 NVIDIA Isaac 官方办公室场景：{office_usd}")
    # ==================== DualVLN 办公室场景：替换粗糙地形（结束） ====================

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    # ==================== DualVLN 办公室场景：固定 Lite3 出生位姿（开始） ====================
    if args_cli.office_scene:
        env_cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (args_cli.spawn_x, args_cli.spawn_x),
            "y": (args_cli.spawn_y, args_cli.spawn_y),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (args_cli.spawn_yaw, args_cli.spawn_yaw),
        }
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    # ==================== DualVLN 办公室场景：固定 Lite3 出生位姿（结束） ====================

    keyboard_controller = None
    ros2_bridge = None
    command_limits = torch.tensor(
        [
            env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        ],
        dtype=torch.float32,
    )
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        keyboard_controller = Se2Keyboard(config)
    # ==================== DualVLN 仿真接入：机器人前视相机（开始） ====================
    if args_cli.ros2_camera:
        if args_cli.camera_rate <= 0.0:
            raise ValueError("--camera_rate must be greater than zero")
        from isaaclab import sim as sim_utils
        from isaaclab.sensors import CameraCfg

        env_cfg.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/TORSO/front_camera",
            update_period=1.0 / args_cli.camera_rate,
            height=384,
            width=384,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.28, 0.0, 0.08),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        # DualVLN 的 System 2 会用“↓”请求向下观察，此相机相对前视相机下俯约 35 度。
        env_cfg.scene.look_down_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/TORSO/look_down_camera",
            update_period=1.0 / args_cli.camera_rate,
            height=384,
            width=384,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.28, 0.0, 0.08),
                rot=(0.6275, -0.3265, 0.3265, -0.6275),
                convention="ros",
            ),
        )
    # ==================== DualVLN 仿真接入：机器人前视相机（结束） ====================

    # ==================== DualVLN 仿真接入：桥接和速度观测（开始） ====================
    if args_cli.ros2_cmd_vel or args_cli.ros2_camera:
        ros2_bridge = Lite3Ros2Bridge(
            command_limits=tuple(command_limits.tolist()),
            subscribe_cmd_vel=args_cli.ros2_cmd_vel,
            publish_camera=args_cli.ros2_camera,
        )
    if args_cli.ros2_cmd_vel:
        env_cfg.commands.base_velocity.debug_vis = False

    if keyboard_controller is not None or args_cli.ros2_cmd_vel:
        def combined_velocity_command(env):
            command = torch.zeros(3, dtype=torch.float32)
            if keyboard_controller is not None:
                command += keyboard_controller.advance().to(dtype=torch.float32, device="cpu")
            if args_cli.ros2_cmd_vel and ros2_bridge is not None:
                command += ros2_bridge.advance()
            command = torch.clamp(command, min=-command_limits, max=command_limits)
            return command.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)

        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=combined_velocity_command,
        )
    # ==================== DualVLN 仿真接入：桥接和速度观测（结束） ====================

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )

    dt = env.unwrapped.step_dt
    # print(dt, "dt")
    # reset environment
    
    obs = env.get_observations()
    
    timestep = 0
    next_camera_publish = 0.0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)
        # ==================== DualVLN 仿真接入：按固定频率发布前视图像（开始） ====================
        if args_cli.ros2_camera and ros2_bridge is not None and time.monotonic() >= next_camera_publish:
            camera = env.unwrapped.scene["front_camera"]
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            intrinsic = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
            look_down_rgb = env.unwrapped.scene["look_down_camera"].data.output["rgb"][0].detach().cpu().numpy()
            ros2_bridge.publish_camera(rgb, intrinsic, look_down_rgb)
            next_camera_publish = time.monotonic() + 1.0 / args_cli.camera_rate
        # ==================== DualVLN 仿真接入：按固定频率发布前视图像（结束） ====================
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard or args_cli.ros2_cmd_vel:
            # 办公室内缩短追踪距离，减少第三人称相机穿墙或被墙遮挡。
            camera_follow(env, distance=1.5 if args_cli.office_scene else 3.0)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if ros2_bridge is not None:
        ros2_bridge.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
