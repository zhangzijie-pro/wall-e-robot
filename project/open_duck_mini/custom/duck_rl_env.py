import torch
from typing import Dict

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import Articulation

# 导入您的机器人配置（假设在同一目录）
from robots.open_duck_mini import OPEN_DUCK_MINI_CFG

class DuckRLEnvCfg(DirectRLEnvCfg):
    # 模拟配置
    sim = SimulationCfg(dt=1 / 120, use_gpu_pipeline=True)  # 高精度物理

    # 场景配置（基于您的NewRobotsSceneCfg）
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)  # 并行128环境

    # 动作和观测空间维度（调整为您的关节数：14个）
    num_actions = 14  # right_5 + left_5 + head/neck_4
    num_observations = 14*2 + 13 + 6 + 2  # joint_pos(14)+vel(14) + root(13) + imu(6: acc3+gyro3) + contacts(2: 左右脚)

    # 其他RL参数
    decimation = 4  # 每4物理步决策一次
    episode_length_s = 20.0  # 每episode 20s

class DuckRLEnv(DirectRLEnv):
    cfg: DuckRLEnvCfg

    def __init__(self, cfg: DuckRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.articulation: Articulation = self.scene["Open_Duck"]  # 您的机器人实例
        self.joint_names = self.articulation.joint_names  # 获取关节名列表
        self.num_joints = len(self.joint_names)

    def _setup_scene(self):
        # 加载地面和灯光（从您的scene_cfg复制）
        self.scene.articulations["Open_Duck"] = OPEN_DUCK_MINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Open_Duck")
        # 添加其他资产如地面（ground = ...）
        super()._setup_scene()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 应用动作：假设动作是关节位置目标
        self.articulation.set_joint_position_target(actions)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        # 核心观测
        joint_pos = self.articulation.data.joint_pos[:, :self.num_joints]
        joint_vel = self.articulation.data.joint_vel[:, :self.num_joints]
        root_state = self.articulation.data.root_state_w  # [pos(3), rot(4), lin_vel(3), ang_vel(3)] = 13

        # 传感器观测（假设已添加）
        imu_data = torch.zeros((self.num_envs, 6), device=self.device)  # acc(3) + gyro(3)
        if "base_imu" in self.articulation.sensors:
            imu = self.articulation.sensors["base_imu"].data
            imu_data[:, :3] = imu.acc  # 加速度
            imu_data[:, 3:] = imu.gyro  # 角速度

        contact_forces = torch.zeros((self.num_envs, 2), device=self.device)  # 左右脚接触（简化到binary）
        if "foot_contact" in self.articulation.sensors:
            contact = self.articulation.sensors["foot_contact"].data
            contact_forces[:, 0] = contact.force_matrix[:, 0, 2] > 1.0  # 右脚Z力 >1 表示接触
            contact_forces[:, 1] = contact.force_matrix[:, 1, 2] > 1.0  # 左脚

        # 组合观测字典（输入到策略网络）
        return {
            "joint_pos": joint_pos,  # [num_envs, 14]
            "joint_vel": joint_vel,  # [num_envs, 14]
            "root_state": root_state,  # [num_envs, 13]
            "imu": imu_data,  # [num_envs, 6]
            "contacts": contact_forces,  # [num_envs, 2]
            # 如果有相机: "rgb": camera_data["rgb"].view(num_envs, height, width, 3)
        }

    def _get_rewards(self) -> torch.Tensor:
        # 奖励：前进速度(根线性速度x) - 能量惩罚(关节速度平方) - 倒地惩罚
        forward_vel = self.articulation.data.root_lin_vel_w[:, 0]  # x方向速度
        energy_penalty = -0.01 * torch.sum(torch.square(self.articulation.data.joint_vel), dim=1)
        upright_bonus = torch.where(torch.abs(self.articulation.data.root_state_w[:, 2] - 0.42) < 0.1, 1.0, -1.0)  # 保持高度
        return forward_vel + energy_penalty + upright_bonus

    def _get_dones(self) -> torch.Tensor:
        # 终止：时间到、倒地(高度<0.3) 或 关节超限
        time_out = self.episode_time >= self.max_episode_length
        died = self.articulation.data.root_state_w[:, 2] < 0.3  # z高度太低
        joint_limit = torch.any(torch.abs(self.articulation.data.joint_pos) > 1.5, dim=1)  # 假设限位±1.5
        return time_out | died | joint_limit

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        # 自定义重置：如您的代码中设置初始状态
        root_state = self.articulation.data.default_root_state.clone()[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.articulation.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        # ... 类似您的重置逻辑