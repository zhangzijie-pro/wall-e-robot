import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
import random

class VoiceIDNode(Node):
    def __init__(self):
        super().__init__("voice_id_node")

        self.subscription = self.create_subscription(
            ByteMultiArray,
            "mic_audio",
            self.listener_callback,
            10
        )


    def listener_callback(self, msg):
        data = msg.data
        if not data:
            return

        # 随机保留三分之一的数据
        reduced_size = len(data) // 3
        selected_indices = sorted(random.sample(range(len(data)), reduced_size))
        self.audio_data = [data[i] for i in selected_indices]
    

def main(args=None):
    rclpy.init(args=args)
    node = VoiceIDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()