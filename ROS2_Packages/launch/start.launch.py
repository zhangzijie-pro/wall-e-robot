from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # slam_gmapping 节点
        Node(
            package='slam_gmapping',
            executable='slam_gmapping',
            name='slam_gmapping',
            output='screen',
            parameters=[
                # 地图分辨率，单位米。建议 0.05 或更小可获得更细地图
                {"delta": 0.05},
                # 地图的最大范围（不影响匹配，仅地图尺寸）
                {"xmin": -10.0},
                {"ymin": -10.0},
                {"xmax": 10.0},
                {"ymax": 10.0},

                # 粒子滤波的粒子数量，数值越大越准确但越慢
                {"particles": 40},

                # 从激光 scan 中使用的最大测距范围（实际用于建图部分）
                {"maxUrange": 8.0},
                # 激光最大量程，如果雷达支持 12m 可以设置此值
                {"maxRange": 12.0},

                # 位移超过此值时更新粒子滤波（单位：米）
                {"linearUpdate": 0.3},
                # 旋转角度超过此值时更新（单位：弧度）
                {"angularUpdate": 0.3},
                # 每经过此时间（秒）强制更新一次地图（不推荐太小）
                {"temporalUpdate": -1.0},  # -1 表示不启用

                # 匹配成功的最小得分，越高越严格（建议：150~300）
                {"minimumScore": 200},

                # 扫描角度间隔最小值，小于此值跳过
                {"sigma": 0.05},
                {"kernelSize": 1},

                # 当粒子重采样的有效权重比例小于此阈值时触发重采样
                {"resampleThreshold": 0.5},

                # 激光扫描频率（Hz），建议和雷达匹配
                {"laserSigma": 0.01},
                {"lstep": 0.05},
                {"astep": 0.05},

                # 地图刷新周期（秒）
                {"map_update_interval": 2.0},

                # 激光使用的 TF frame 名称
                {"base_frame": "base_link"},
                {"odom_frame": "odom"},
                {"map_frame": "map"},
                {"throttle_scans": 1},  # 每帧都处理

                # 开启此参数将运行改进的运动模型
                {"srr": 0.1},  # 旋转时位移误差
                {"srt": 0.2},  # 旋转时角度误差
                {"str": 0.1},  # 平移时角度误差
                {"stt": 0.2},  # 平移时位移误差

                # 初始化地图偏移（不建议更改）
                {"xmin": -100.0},
                {"ymin": -100.0},
                {"xmax": 100.0},
                {"ymax": 100.0},

                # 可选 debug 输出
                {"tf_delay": 0.05},  # tf 延迟补偿
                {"transform_publish_period": 0.05},  # tf 发布频率
                {"occ_thresh": 0.25},  # 占用概率阈值
                {"llsamplerange": 0.01},
                {"llsamplestep": 0.01},
                {"lasamplerange": 0.005},
                {"lasamplestep": 0.005}
            ]
        ),

        # 静态 TF: odom → base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
        ),

        # 静态 TF: base_link → laser
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_laser_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'laser']
        )
    ])
