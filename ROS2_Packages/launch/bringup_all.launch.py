from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    def include(name):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    FindPackageShare('ROS2_Packages').find('ROS2_Packages'),
                    'launch',
                    f'{name}.launch.py'
                )
            )
        )
    
    return LaunchDescription([
        include('perception'),
        include('cognition'),
        include('expression'),
        include('motion'),
        include('mapping'),
    ])
