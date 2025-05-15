from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='camera_node', executable='camera_node', name='camera_node', output='screen'),
        Node(package='face_recognition_node', executable='face_recognition_node', name='face_recognition_node', output='screen'),
        Node(package='mic_audio_node', executable='mic_audio_node', name='mic_audio_node', output='screen'),
        Node(package='voice_id_node', executable='voice_id_node', name='voice_id_node', output='screen'),
        Node(package='vosk_stt_node', executable='vosk_stt_node', name='vosk_stt_node', output='screen'),
    ])
