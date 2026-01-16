# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from robots.open_duck_mini import OPEN_DUCK_MINI_CFG

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Open_Duck = OPEN_DUCK_MINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Open_Duck")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()  # 获取物理时间步长
    sim_time = 0.0  # 初始化仿真时间
    count = 0  # 计数器，用于控制行为逻辑

    while simulation_app.is_running():  # 当仿真运行时持续进行
        # 每500个计数重置一次状态
        if count % 500 == 0:
            count = 0  # 重置计数器

            # 获取初始根状态并根据环境的原点进行偏移
            root_dofbot_state = scene["Open_Duck"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins

            # 传递根状态给仿真系统，更新机器人的位置和速度
            scene["Open_Duck"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Open_Duck"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            # 传递初始关节状态给仿真系统
            joint_pos, joint_vel = (
                scene["Open_Duck"].data.default_joint_pos.clone(),
                scene["Open_Duck"].data.default_joint_vel.clone(),
            )
            scene["Open_Duck"].write_joint_state_to_sim(joint_pos, joint_vel)

            # 清理内部缓存，准备进行下一轮仿真
            scene.reset()
            print("[INFO]: Resetting Open_Duck state...")

        # 控制机器人直行
        if count % 100 < 75:
            action = torch.Tensor([[20.0, 15.0]])  # 向前驱动
        else:
            # 控制机器人转弯
            action = torch.Tensor([[5.0, -15.0]])  # 向左转弯

        # 将动作发送给机器人关节
        # scene["Open_Duck"].set_joint_velocity_target(action)

        # # 进行波动动作
        wave_action = scene["Open_Duck"].data.default_joint_pos
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)  # 基于时间周期变化
        scene["Open_Duck"].set_joint_position_target(wave_action)

        # 更新场景数据并写入仿真
        scene.write_data_to_sim()
        sim.step()  # 执行一步仿真
        sim_time += sim_dt  # 增加仿真时间
        count += 1  # 更新计数器

        # 更新场景并根据时间步长更新状态
        scene.update(sim_dt)

        if count % 50 == 0:
            print(f"[INFO]: Sim Time: {sim_time:.2f}, Action: {action.numpy()}")


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
