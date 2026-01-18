"""Configuration for the Open Duck Mini robot in Isaac Lab."""

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import FrameTransformerCfg, CameraCfg

OPEN_DUCK_MINI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Open_Duck",  # 多环境时自动替换
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:\\Nvidia_Sim\\project\\assets\\Robots\\open_duck_mini\\open_duck_mini.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # max_joint_velocity=1000.0,
            enable_gyroscopic_forces=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 关闭自碰撞（四足机器人必需）
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),  # 初始高度（Open Duck ~42cm，根据 URDF 调整防塌陷）
        joint_pos={
            # Right leg joints (从 URDF)
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.2,  # 轻微弯曲以稳定
            "right_knee": -0.5,
            "right_ankle": 0.3,
            
            "neck_pitch": 0.0,
            "head_pitch": 0.3,
            "head_yaw" : 0.1,
            "head_roll": 0.0,
            

            # Left leg joints (对称，从 URDF 推断；如果 URDF 只显示 right，复制 left)
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.2,
            "left_knee": -0.5,
            "left_ankle": 0.3,
            
        },
        joint_vel={jname: 0.0 for jname in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll"
        ]},
    ),
    soft_joint_pos_limit_factor=0.95,  # 关节限位安全裕度
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", ".*_ankle", "neck_.*", "head_.*"# 正则匹配所有腿关节
            ],
            effort_limit=300.0,  # 力矩限（根据伺服调整）
            velocity_limit=50.0,  # 速度限
            stiffness=0.0,  # Velocity 控制模式（稳定）
            damping=20000.0,  # 高阻尼防抖动；如果爆炸，提高到 5e4
        ),
    },
)