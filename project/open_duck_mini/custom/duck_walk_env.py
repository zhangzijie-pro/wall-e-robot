import torch
import pickle  # 用于加载参考运动
import numpy as np
from typing import Dict

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg, flat_patch_terrain, rough_terrain
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation

# 您的机器人CFG
from robots.open_duck_mini import OPEN_DUCK_MINI_CFG

class DuckWalkEnvCfg(DirectRLEnvCfg):
    sim = SimulationCfg(dt=1/200.0, use_gpu_pipeline=True, gravity=(0.0, 0.0, -9.81))  # 高精度，匹配MuJoCo
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=True)  # 并行256环境
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="plane",  # 默认平坦；任务中切换
        terrain_generator=flat_patch_terrain,  # 或rough_terrain for rough
    )
    num_actions = 14  # 您的关节数
    num_observations = 14*2 + 13 + 6 + 2  # joint_pos/vel + root + imu + contacts
    decimation = 4
    episode_length_s = 20.0

class DuckWalkEnv(DirectRLEnv):
    cfg: DuckWalkEnvCfg
    use_imitation = True  # 设为False如果无参考运动
    reference_coeffs = None  # 参考多项式系数

    def __init__(self, cfg: DuckWalkEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.articulation: Articulation = self.scene["Open_Duck"]
        self.num_joints = 14  # 匹配您的关节
        self.joint_indices = {name: idx for idx, name in enumerate(self.articulation.joint_names)}

        # 加载参考运动（从原仓库生成）
        if self.use_imitation:
            with open("path/to/polynomial_coefficients.pkl", "rb") as f:  # 替换为您的路径
                self.reference_coeffs = pickle.load(f)  # 假设dict of coeffs per joint

        # 切换地形基于任务（e.g., flat or rough）
        if "rough" in cfg.task_name:  # 通过CLI传递task
            self.cfg.terrain.terrain_generator = rough_terrain

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.articulation.set_joint_position_target(actions)  # 位置控制，匹配原政策

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        joint_pos = self.articulation.data.joint_pos[:, :self.num_joints]
        joint_vel = self.articulation.data.joint_vel[:, :self.num_joints]
        root_state = self.articulation.data.root_state_w  # [pos3, rot4, lin_vel3, ang_vel3]

        imu_data = torch.zeros((self.num_envs, 6), device=self.device)
        if "base_imu" in self.articulation.sensors:
            imu = self.articulation.sensors["base_imu"].data
            imu_data[:, :3] = imu.acc
            imu_data[:, 3:] = imu.gyro

        contacts = torch.zeros((self.num_envs, 2), device=self.device)
        if "foot_contact" in self.articulation.sensors:
            contact = self.articulation.sensors["foot_contact"].data
            contacts[:, 0] = contact.force_matrix[:, self.joint_indices["right_ankle"], 2] > 1.0  # 右脚Z力
            contacts[:, 1] = contact.force_matrix[:, self.joint_indices["left_ankle"], 2] > 1.0

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_state": root_state,
            "imu": imu_data,
            "contacts": contacts,
        }

    def _get_rewards(self) -> torch.Tensor:
        forward_vel = self.articulation.data.root_lin_vel_w[:, 0]
        energy_penalty = -0.01 * torch.sum(torch.square(self.articulation.data.joint_vel), dim=1)
        upright_bonus = torch.where(torch.abs(self.articulation.data.root_state_w[:, 2] - 0.42) < 0.1, 1.0, -1.0)

        imitation_reward = torch.zeros(self.num_envs, device=self.device)
        if self.use_imitation:
            # 计算参考关节位置（基于时间的多项式）
            t = self.episode_time % 2.0  # 假设周期2s；调整
            ref_pos = torch.zeros_like(self.articulation.data.joint_pos)
            for joint, coeffs in self.reference_coeffs.items():
                idx = self.joint_indices.get(joint, -1)
                if idx != -1:
                    ref_pos[:, idx] = np.polyval(coeffs, t)  # 多项式求值
            imitation_reward = -torch.mean(torch.square(self.articulation.data.joint_pos - ref_pos), dim=1)

        return forward_vel + energy_penalty + upright_bonus + 0.5 * imitation_reward  # 权重调整

    def _get_dones(self) -> torch.Tensor:
        time_out = self.episode_time >= self.max_episode_length
        died = self.articulation.data.root_state_w[:, 2] < 0.3
        return time_out | died

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        # 随机化初始状态（domain rand）
        self.articulation.apply_randomization(env_ids)
        # 设置初始位置等，如您的原脚本