import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from std_msgs.msg import String
from sensor_msgs.msg import Image
import random
from utils.msg import AudioStream
from shared_utils.recognize import SpeakerRecognize_Data, SpeakerRecognize

class VoiceIDNode(Node):
    def __init__(self):
        super().__init__("voice_id_node")

        self.min_confidence = 0.6
        self.SRData = SpeakerRecognize_Data()
        self.subscription_audio = self.create_subscription(
            ByteMultiArray,
            "raw_audio",
            self.audio_callback,
            10
        )
        self.subscription_image = self.create_subscription(
            Image,
            "image_raw",
            self.image_callback,
            10
        )

        self.publisher_voice_id = self.create_publisher(
            String,
            "speaker_id",
            self.speaker_id            
        )

    def audio_callback(self, msg):
        data = msg.data
        if not data:
            return

        # 随机保留三分之一的数据
        reduced_size = len(data) // 3
        selected_indices = sorted(random.sample(range(len(data)), reduced_size))
        audio_data = [data[i] for i in selected_indices]
        self.SRData._set_audio_tensor(audio_data)

    def image_callback(self, msg):
        data = msg.data
        if not data:
            return
    
    def get_speaker_id(self):
        sr = SpeakerRecognize(self.SRData)
        face_ids, voice_id = sr.Start_()
        names = [k for name in face_ids for k,_ in name.items() ]
        if not face_ids:
            return voice_id
        
        if voice_id[0] in names and voice_id[1] > self.min_confidence:
            return voice_id
        
        if voice_id[1] < self.min_confidence and voice_id[0] not in names:
            if len(names) == 1:
                return names[0]
            else:
                score = {}
                for name in names:
                    score[name] = 1.0
                    if name == voice_id[0]:
                        score[name] += voice_id[1]
                return max(score, key=score.get)
            
        return voice_id[0] if voice_id[0] else names[0]

    def speaker_id(self):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = VoiceIDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()