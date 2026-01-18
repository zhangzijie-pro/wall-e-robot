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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns

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
    
    sensors = {
        "base_frame": FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Open_Duck/base_link",  # 假设base_link是根
            target_prim_path="{ENV_REGEX_NS}/Robot",  # 目标帧
        ),
        "camera": CameraCfg(
            prim_path="{ENV_REGEX_NS}/Open_Duck/head_camera",  # 假设头部有相机挂点
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "depth"],
            width=640, height=480,
        ),
        "height_scanner": RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Open_Duck/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=True,
            mesh_prim_paths=["/World/defaultGroundPlane"],
        ),
        "contact_forces":ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Open_Duck/.*_ankle", update_period=0.0, history_length=6, debug_vis=True
        )
    }
    
    Open_Duck.sensors = sensors


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()  # 获取物理时间步长
    sim_time = 0.0  # 初始化仿真时间
    count = 0  # 计数器，用于控制行为逻辑
    
    kp = 50.0  # 位置增益
    kd = 5.0   # 速度增益

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
        # if count % 100 < 75:
        #     action = torch.Tensor([[20.0, 15.0]])  # 向前驱动
        # else:
        #     # 控制机器人转弯
        #     action = torch.Tensor([[5.0, -15.0]])  # 向左转弯

        # 简单正弦步态（假设双腿简化）
        phase = 2 * np.pi * 0.5 * sim_time
        right_leg_pos = torch.tensor([0.0, 0.0, -0.2 + 0.1 * np.sin(phase), -0.5 + 0.1 * np.cos(phase), 0.3])
        left_leg_pos = torch.tensor([0.0, 0.0, -0.2 + 0.1 * np.sin(phase + np.pi), -0.5 + 0.1 * np.cos(phase + np.pi), 0.3])
        head_pos = torch.tensor([0.0, 0.3, 0.1 * np.sin(phase), 0.0])  # 头部摆动
        full_target = torch.cat((right_leg_pos, head_pos, left_leg_pos)).unsqueeze(0)  # 匹配关节顺序
        # scene["Open_Duck"].set_joint_position_target(full_target)
        
        target_pos = full_target
        current_pos = scene["Open_Duck"].data.joint_pos
        current_vel = scene["Open_Duck"].data.joint_vel
        efforts = kp * (target_pos - current_pos) - kd * current_vel
        scene["Open_Duck"].set_joint_effort_target(efforts)
        scene["Open_Duck"].set_joint_position_target(target_pos)  # 混合使用
        # 进行波动动作
        # wave_action = scene["Open_Duck"].data.default_joint_pos
        # wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)  # 基于时间周期变化
        # scene["Open_Duck"].set_joint_position_target(wave_action)

        # 更新场景数据并写入仿真
        scene.write_data_to_sim()
        sim.step()  # 执行一步仿真
        sim_time += sim_dt  # 增加仿真时间
        count += 1  # 更新计数器

        # 更新场景并根据时间步长更新状态
        scene.update(sim_dt)

        if count % 50 == 0:
            print(f"[INFO]: Sim Time: {sim_time:.2f}, Action: {action.numpy()}")
            print("-------------------------------")
            print(scene["Open_Duck"].get_sensor_data("camera"))
            print("Received shape of rgb   image: ", scene["Open_Duck"].get_sensor_data("camera").data.output["rgb"].shape)
            print("Received shape of depth image: ", scene["Open_Duck"].get_sensor_data("camera").data.output["depth"].shape)
            print("-------------------------------")
            print(scene["Open_Duck"].get_sensor_data("height_scanner"))
            print("Received max height value: ", torch.max(scene["Open_Duck"].get_sensor_data("height_scanner").data.ray_hits_w[..., -1]).item())
            print("-------------------------------")
            print(scene["Open_Duck"].get_sensor_data("contact_forces"))
            print("Received max contact force of: ", torch.max(scene["Open_Duck"].get_sensor_data("contact_forces").data.net_forces_w).item())



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
