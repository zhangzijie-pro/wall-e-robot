# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from open_duck_mini.robots.open_duck_mini import OPEN_DUCK_MINI_CFG

@configclass
class OpenDuckMiniEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 14
    observation_space = 14*2 + 13 + 6 + 2
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg: ArticulationCfg = OPEN_DUCK_MINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Open_Duck")

    # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,                   # 先从小开始调试，训练时可加大到256~1024
        env_spacing=2.5,
        replicate_physics=True,
    )

    # custom parameters/scales
    # - controllable joint
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"
    
    # - action scale
    action_scale = 1.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    
    rew_scale_forward = 1.0             # 前进速度奖励
    rew_scale_energy = -0.01            # 能量惩罚（关节速度平方）
    rew_scale_upright = 0.5             # 保持直立bonus
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
    max_root_angle = 0.5