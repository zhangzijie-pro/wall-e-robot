# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .open_duck_mini_env_cfg import OpenDuckMiniEnvCfg

class OpenDuckMiniEnv(DirectRLEnv):
    cfg: OpenDuckMiniEnvCfg

    def __init__(self, cfg: OpenDuckMiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = None

        self.joint_pos = None
        self.joint_vel = None

    def _setup_scene(self):
        # 创建机器人实例
        self.robot = Articulation(self.cfg.robot_cfg)

        # 加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 克隆多环境
        self.scene.clone_environments(copy_from_source=False)

        # CPU模式下过滤碰撞（可选）
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 注册到scene
        self.scene.articulations["Open_Duck"] = self.robot

        # 加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 简单位置目标控制（假设你的actuators是position-based，可改成effort）
        scaled_actions = actions * self.cfg.action_scale
        self.robot.set_joint_position_target(scaled_actions)  # 全关节控制

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        obs = torch.cat(
            [
                self.joint_pos,                     # [num_envs, 14]
                self.joint_vel,                     # [num_envs, 14]
                self.robot.data.root_state_w,       # [num_envs, 13] pos+rot+lin_vel+ang_vel
                # 示例：如果有IMU和contact，可加
                # self.robot.data.imu_acc,          # 假设有
                # self.robot.data.contact_forces.mean(dim=1),  # 简化
            ],
            dim=-1,
        )

        return {"policy": obs}  # 策略观测（可加"critic"用于不对称AC）

    def _get_rewards(self) -> torch.Tensor:
        # 前进速度（x方向）
        forward_vel = self.robot.data.root_lin_vel_w[:, 0]

        # 能量惩罚
        energy_penalty = self.cfg.rew_scale_energy * torch.sum(torch.square(self.joint_vel), dim=1)

        # 直立bonus（z高度接近初始）
        upright = self.cfg.rew_scale_upright * (1.0 - torch.abs(self.robot.data.root_state_w[:, 2] - 0.42) / 0.1)

        # 存活奖励
        alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())

        total_reward = forward_vel + energy_penalty + upright + alive

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos  # 更新

        # 超时
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        # 倒地（根z太低 或 倾斜太大）
        died = self.robot.data.root_state_w[:, 2] < 0.25
        tilted = torch.norm(self.robot.data.root_state_w[:, 3:7], dim=1) > math.cos(self.cfg.max_root_angle)  # 四元数w < cos(theta/2)

        terminated = died | tilted

        return terminated, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 初始关节位置（从default + 小噪声）
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos += sample_uniform(-0.05, 0.05, joint_pos.shape, joint_pos.device)  # 小扰动

        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # 根姿态：高度随机 + 位置偏移
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 2] += sample_uniform(  # z高度
            self.cfg.initial_height_range[0],
            self.cfg.initial_height_range[1],
            (len(env_ids),),
            root_state.device,
        )
        root_state[:, :3] += self.scene.env_origins[env_ids]  # env origin偏移

        # 写入仿真
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 更新缓存
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel