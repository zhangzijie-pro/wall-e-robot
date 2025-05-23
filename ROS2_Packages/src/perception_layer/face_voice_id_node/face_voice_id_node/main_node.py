import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from std_msgs.msg import String
from sensor_msgs.msg import Image
import random
from utils.msg import AudioStream
from shared_utils.audio_helper import play_beep

class VoiceIDNode(Node):
    def __init__(self):
        super().__init__("voice_id_node")

        self.subscription_audio = self.create_subscription(
            ByteMultiArray,
            "raw_audio",
            self.listener_callback,
            10
        )
        self.subscription_image = self.create_subscription(
            Image,
            "image_raw",
            self.listener_callback,
            10
        )

        self.publisher_voice_id = self.create_publisher(
            String,
            "speaker_id",
            self.speaker_id            
        )

    def listener_callback(self, msg):
        data = msg.data
        if not data:
            return

        # 随机保留三分之一的数据
        reduced_size = len(data) // 3
        selected_indices = sorted(random.sample(range(len(data)), reduced_size))
        self.audio_data = [data[i] for i in selected_indices]
    
    def speaker_id(self):
        pass
    

def main(args=None):
    rclpy.init(args=args)
    node = VoiceIDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()