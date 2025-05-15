from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='executor_node', executable='executor_node', name='executor_node', output='screen'),
        Node(package='langchain_agent_node', executable='langchain_agent_node', name='langchain_agent_node', output='screen'),
        Node(package='memory_manager_node', executable='memory_manager_node', name='memory_manager_node', output='screen'),
        Node(package='task_planner_node', executable='task_planner_node', name='task_planner_node', output='screen'),

    ])
