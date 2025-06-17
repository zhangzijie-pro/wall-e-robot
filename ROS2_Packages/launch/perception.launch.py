from launch import LaunchDescription
from launch_ros.actions import Node
import yaml
import os

def generate_launch_description():
    config_path = os.path.join(os.getcwd(), "config","config.yaml")
    yaml_component = yaml.safe_load(open(config_path, "r"))

    camera_node = yaml_component["camera_node"]
    mic_audio_node = yaml_component["mic_audio_node"]
    stt_node = yaml_component["stt_node"]


    return LaunchDescription([
        Node(
            package='camera_node', 
            executable='camera_node', 
            name='camera_node', 
            parameters=[{
                'camera_type':camera_node["camera_type"],
                'fps':camera_node["fps"],
                "resolution_H":camera_node["resolution_H"],
                "resolution_W":camera_node["resolution_W"],
            }],
            output='screen'
        ),
        Node(package='face_recognition_node', executable='face_recognition_node', name='face_recognition_node', output='screen'),
        Node(
            package='mic_audio_node', 
            executable='mic_audio_node', 
            name='mic_audio_node', 
            parameters=[{
                'baud_rate':mic_audio_node["baud_rate"],
                'block_size':mic_audio_node["block_size"],
                "sample_rate":mic_audio_node["sample_rate"],
                "channels":mic_audio_node["channels"],
            }],
            output='screen'),
        Node(
            package='vosk_stt_node', 
            executable='vosk_stt_node', 
            name='vosk_stt_node', 
            parameters=[{
                'model_path':stt_node["model_path"],
                'block_size':stt_node["block_size"],
                "buffer_size":stt_node["buffer_size"],
                "queue_size":stt_node["queue_size"],
                "audio_topic":stt_node["audio_topic"],
                "text_topic":stt_node["text_topic"],
                "silence_threshold":stt_node["silence_threshold"],
                "silence_duration":stt_node["silence_duration"],
            }],
            output='screen'),
    ])
