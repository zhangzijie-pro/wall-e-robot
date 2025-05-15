from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='lidar_node', executable='lidar_node', name='lidar_node', output='screen'),
        Node(package='nav2_bt_navigator', executable='nav2_bt_navigator', name='nav2_bt_navigator', output='screen'),
        Node(package='slam_toolbox_node', executable='slam_toolbox_node', name='slam_toolbox_node', output='screen'),
    ])
