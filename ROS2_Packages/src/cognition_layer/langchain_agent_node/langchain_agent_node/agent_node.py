import rclpy
from rclpy.node import Node
from std_msgs.msg import Agent_msg
import yaml
import os
import base64
ROOT_PATH = os.getcwd()


class LangchainAgentNode(Node):
    def __init__(self):
        super().__init__('cognition_layer/agent_node')
        self.subscriptions_ = self.create_subscription(
            Agent_msg,
            'cognition_layer/agent_msg',
            self.agent_msg_callback,
            10
        )

    def agent_msg_callback(self, msg):
        """
        text: String
        speaker_id: int
        img: float32[]
        """
        text_msg = msg.text
        speaker_id = msg.speaker_id
        img = msg.img
        name, history_path = self._speaker(speaker_id)
        
        prompt = {
            'text': text_msg,
            'img': img,
            'name': name,
            'history_path': history_path
        }


    def _speaker(self, speaker_id):
        """
        Returns speaker and his/her history
        """
        config_path = os.path.join(ROOT_PATH, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        return config['agent_node']['speaker'][speaker_id], os.path.join(ROOT_PATH, 'config', config['agent_node']['speaker_history'][speaker_id])
