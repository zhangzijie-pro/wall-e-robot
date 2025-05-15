from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='motion_controller_node', executable='motion_controller_node', name='motion_controller_node', output='screen'),
        Node(package='servo_controller_node', executable='servo_controller_node', name='servo_controller_node', output='screen'),
        Node(package='esp32_comm_node', executable='esp32_comm_node', name='esp32_comm_node', output='screen'),
    ])
