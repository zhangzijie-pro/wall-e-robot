
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # add_robot.py 
    sim_dt = sim.get_physics_dt()  # 获取物理时间步长
    sim_time = 0.0  # 初始化仿真时间
    count = 0  # 计数器，用于控制行为逻辑

    while simulation_app.is_running():  # 当仿真运行时持续进行
        # ... 您的现有重置和动作逻辑 ...

        # 新添加：读取传感器数据（在update前）
        if "head_camera" in scene["Open_Duck"].sensors:
            camera_data = scene["Open_Duck"].sensors["head_camera"].data
            if camera_data["rgb"] is not None:  # 检查数据是否可用
                print(f"[INFO] Camera RGB shape: {camera_data['rgb'].shape}, Depth min: {camera_data['depth'].min()}")

        if "base_imu" in scene["Open_Duck"].sensors:
            imu_data = scene["Open_Duck"].sensors["base_imu"].data
            print(f"[INFO] IMU Acceleration: {imu_data.acc[0].numpy()}")  # 第一环境的加速度

        if "foot_contact" in scene["Open_Duck"].sensors:
            contact_data = scene["Open_Duck"].sensors["foot_contact"].data
            print(f"[INFO] Foot Contact Forces: {contact_data.force_matrix[0].numpy()}")  # 接触力矩阵

        # 更新场景数据并写入仿真
        scene.write_data_to_sim()
        sim.step()  # 执行一步仿真
        sim_time += sim_dt  # 增加仿真时间
        count += 1  # 更新计数器

        # 更新场景并根据时间步长更新状态（这会刷新传感器数据）
        scene.update(sim_dt)

        if count % 50 == 0:
            print(f"[INFO]: Sim Time: {sim_time:.2f}, Action: {action.numpy()}")
            
# CFG 
# 添加域随机化（sim2real gap）
from isaaclab.utils.noise import UniformNoiseCfg, GaussianNoiseCfg

OPEN_DUCK_MINI_CFG.randomization = {
    "joint_pos": UniformNoiseCfg(min=-0.05, max=0.05),  # 初始位置噪声
    "joint_vel": GaussianNoiseCfg(mean=0.0, std=0.1),  # 速度噪声
    "actuator_damping": UniformNoiseCfg(min=15000.0, max=25000.0),  # 匹配MuJoCo damping
    # 添加更多如mass, friction如果需要
}