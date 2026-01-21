# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import numpy as np

from isaaclab.envs import ViewerCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from gymnasium import spaces

# from isaaclab_tasks.direct.open_duck.open_duck_mini import OPEN_DUCK_MINI_CFG
"""Configuration for the Open Duck Mini robot in Isaac Lab."""

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import FrameTransformerCfg, CameraCfg

OPEN_DUCK_MINI_CFG = ArticulationCfg(
    prim_path="/World/envs/env_0/Robot",  # 多环境时自动替换
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:\\Nvidia_Sim\\project\\assets\\Robots\\open_duck_mini\\open_duck_mini.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.5,  # 增加线性阻尼以减少抽搐
            angular_damping=0.5,  # 增加角阻尼以提高稳定性
            max_linear_velocity=5.0,
            enable_gyroscopic_forces=True,
            # disable_gravity=False,  # 重力已默认启用，无需额外设置
            max_depenetration_velocity=10.0,  # 增加去渗透速度以处理接触碰撞
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # enabled_self_collisions=False,  # 关闭自碰撞以避免内部抖动（如果模型有重叠部分）
            solver_position_iteration_count=32,  # 增加迭代次数以改善关节稳定性
            solver_velocity_iteration_count=12,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # 略微提高初始高度，避免落地冲击
        joint_pos={
            "right_hip_yaw":   0.0,
            "right_hip_roll":  0.0,   # 无外展，保持中立
            "right_hip_pitch": -0.3,  # 髋前倾以降低重心
            "right_knee":      -0.6,  # 膝盖弯曲以蹲姿增加稳定性
            "right_ankle":     0.3,   # 踝关节调整以匹配膝盖弯曲，确保脚掌接触地面

            # 左腿：完全对称
            "left_hip_yaw":    0.0,
            "left_hip_roll":   0.0,
            "left_hip_pitch":  -0.3,
            "left_knee":       -0.6,
            "left_ankle":      0.3,

            # 头部：保持中立
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw":   0.0,
            "head_roll":  0.0,
        },
        joint_vel={jname: 0.0 for jname in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll"
        ]},
    ),
    soft_joint_pos_limit_factor=0.98,  # 略微增加限位裕度以避免限位抖动
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee", ".*_ankle"],
            effort_limit=120.0,  # 增加力矩限以处理重力
            velocity_limit=20.0,
            stiffness=150.0,  # 增加刚度以改善响应
            damping=15.0,     # 增加阻尼以减少振荡
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["neck_.*", "head_.*"],
            stiffness=50.0,
            damping=5.0,
            effort_limit=80.0,
        )
    },
)


@ configclass
class OpenDuckEnvCfg(DirectRLEnvCfg):
    # =========================
    # 基础
    # =========================
    decimation: int = 4
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False

    # =========================
    # 仿真 & Viewer
    # =========================
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1280, 720),
    )

    # =========================
    # Scene
    # =========================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=3.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    # =========================
    # Robot
    # =========================
    robot: ArticulationCfg = OPEN_DUCK_MINI_CFG.replace(prim_path="/World/envs/env_0/Robot")

    # =========================
    # Action / Observation
    # =========================
    # num_actions: int = 14
    # num_observations: int = 34

    # action_space = gym.spaces.Box(
    #     low=-1.0,
    #     high=1.0,
    #     shape=(14,),
    #     dtype=np.float32,
    # )
    
    # observation_space = gym.spaces.Dict(
    #     {
    #         "policy": gym.spaces.Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(34,),
    #             dtype=np.float32,
    #         )
    #     }
    # )
    
    # state_space = gym.spaces.Box(
    #     low=-np.inf,
    #     high=np.inf,
    #     shape=(1,),
    #     dtype=np.float32,
    # )

    # =========================
    # Action / Observation
    # =========================
    num_actions: int = 14
    num_observations: int = 34

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(14,),
        dtype=np.float32,
    )
    
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(34,),
        dtype=np.float32,
    )
    
    state_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(34,),
        dtype=np.float32,
    )

    # =========================
    # Reward 参数
    # =========================
    action_scale: float = 0.5
    lin_vel_scale: float = 1.0
    energy_cost_scale: float = 0.05
    alive_reward_scale: float = 1.0
    termination_height: float = 0.25
    max_root_angle: float = 1.0  # radians