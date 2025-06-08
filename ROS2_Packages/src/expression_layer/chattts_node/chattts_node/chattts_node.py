import ChatTTS
from stream import ChatStreamer
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray


class ChatTTSNode(Node):
    def __init__(self):
        super().__init__('chat_tts_node')
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)

        self.rand_spk = self.chat.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.rand_spk,  # add sampled speaker
            temperature=0.1,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )

        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_4]',
            #spk_emb=self.rand_spk,  # add sampled speaker
            temperature=0.1,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )
        self.subscription_audio = self.create_subscription(
            ByteMultiArray,
            "text_audio",
            self.audio_callback,
            10
        )
    def audio_callback(self, msg):
        data = msg.data
        if not data:
            return

        text = data.decode('utf-8')
        streamchat = self._infer(text)
        self.get_logger().info(f"Received text: {text}")
        

    def _infer(self, text):
        streamchat = self.chat.infer(
            [],
            stream=True,
            params_infer_code=self.params_infer_code,
            params_refine_text=self.params_refine_text
        )
        return streamchat

# ChatStreamer().play(streamchat)
