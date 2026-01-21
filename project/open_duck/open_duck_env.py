# open_duck_mini_env_fixed.py

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .open_duck_env_cfg import OPEN_DUCK_MINI_CFG, OpenDuckEnvCfg


class OpenDuckMiniEnv(DirectRLEnv):
    cfg: OpenDuckEnvCfg

    def __init__(self, cfg: OpenDuckEnvCfg, render_mode: str | None = None, **kwargs):
        # 初始化动作缓存
        self.actions: torch.Tensor | None = None
        
        # 这里的 super().__init__ 会自动调用 _setup_scene
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # 1. 首先实例化 Robot
        self.robot = Articulation(self.cfg.robot)

        # 2. 添加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 3. 注册到 Scene 之前需要 clone (通常 DirectRLEnv 会处理)
        # 注意：DirectRLEnv 的流程中，你需要先定义对象，然后 clone，最后注册到 scene
        self.scene.clone_environments(copy_from_source=False)

        # 4. 注册 Articulation 到 Scene
        self.scene.articulations["robot"] = self.robot

        # 5. 其他设置
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        if self.robot is None:
            raise RuntimeError("Robot not initialized in _pre_physics_step!")
        self.actions = actions.clone()
        # Δ joint position control
        targets = self.robot.data.default_joint_pos + actions * self.cfg.action_scale
        self.robot.set_joint_position_target(targets)
    # ---------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------
    def _apply_action(self):
        if self.robot is None:
            raise RuntimeError("Robot not initialized in _apply_action!")
        actions = self.actions
        
        self.actions = actions.clone()
        
        # 计算关节目标位置: 默认姿态 + (动作 * 缩放)
        targets = self.robot.data.default_joint_pos + actions * self.cfg.action_scale
        
        self.robot.set_joint_position_target(targets)
        
    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        if self.robot is None:
            raise RuntimeError("Robot not initialized in _get_observations!")
        obs = torch.cat(
            [
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.robot.data.root_lin_vel_w,
                self.robot.data.root_ang_vel_w,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        forward_vel = self.robot.data.root_lin_vel_w[:, 0]
        quat_w = self.robot.data.root_state_w[:, 6]
        upright = quat_w
        energy = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        reward = 1.0 + 1.0 * forward_vel + 1.5 * upright - 0.0005 * energy
        return reward

    # ---------------------------------------------------------------------
    # Dones
    # ---------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        height = self.robot.data.root_state_w[:, 2]
        died = height < self.cfg.termination_height
        quat_w = self.robot.data.root_state_w[:, 6]
        tilted = quat_w < math.cos(self.cfg.max_root_angle / 2)
        terminated = died | tilted
        return terminated, timeout

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------
    # def _reset_idx(self, env_ids: torch.Tensor | None):
    #     if env_ids is None:
    #         env_ids = torch.arange(self.scene.num_envs, device=self.device)
    #     # 重置 robot 数据
    #     self.robot.reset(env_ids)
    #     super()._reset_idx(env_ids)

    #     # 使用默认策略重置状态
    #     root_state = self.robot.data.default_root_state[env_ids].clone()
    #     root_state[:, :3] += self.scene.env_origins[env_ids]
    #     joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    #     joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

    #     # 写入到仿真
    #     self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    #     self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
    #     self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 1. 必须先调用 robot 的 reset
        self.robot.reset(env_ids)
        
        # 2. 调用父类的 reset (处理 episode_length_buf 等)
        super()._reset_idx(env_ids)

        # 3. 获取默认状态并确保设备一致
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # 加上环境偏移
        root_state[:, :3] += self.scene.env_origins[env_ids]
        
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 4. 写入仿真 (注意：必须在 super()._reset_idx 之后或按需执行)
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # self.scene._update_full_tensor_views()
