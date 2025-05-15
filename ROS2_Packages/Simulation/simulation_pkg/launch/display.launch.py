import os
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import launch_ros.parameter_descriptions
def generate_launch_description():
    # 获取默认的urdf路径
    urdf_package_path = get_package_share_directory("robot_description")
    urdf_path = os.path.join(urdf_package_path,'urdf','first_robot.urdf')
    default_rviz_config_path = os.path.join(urdf_package_path,'config','display_robot_model.rivz')
    
    action_declare_arg_mode_path = launch.actions.DeclareLaunchArgument(
        name='model',default_value=str(urdf_path),description="加载模型文件路径"
    )
    # 通过文件路径, 获取内容, 并转换参数对象,供给robot_state_publisher
    subsitions_command_result = launch.substitutions.Command(['xacro ',
    launch.substitutions.LaunchConfiguration('model')])
    robot_description_value = launch_ros.parameter_descriptions\
        .ParameterValue(subsitions_command_result,value_type=str)
    action_robot_state_publisher = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description':robot_description_value}]
    )
    action_joint_state_publisher = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
    )
    action_rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=["-d",default_rviz_config_path]
    )
    
    return launch.LaunchDescription([
        action_declare_arg_mode_path,
        action_robot_state_publisher,
        action_joint_state_publisher,
        action_rviz_node,
    ])