from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='chattts_node', executable='chattts_node', name='chattts_node', output='screen'),
        Node(package='audio_play_client', executable='audio_play_client', name='audio_play_client', output='screen'),
    ])
