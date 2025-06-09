import rclpy
from rclpy import Node
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import String
import ChatTTS
from stream import ChatStreamer

class TTS:
    """
    ChatTTS
    """
    def __init__(self):
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)

        self.datas = None
        self.step = 8
        rand_spk = self.chat.sample_random_speaker()
        
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,  # add sampled speaker
            temperature=0.1,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )

        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_4]',
            #spk_emb=rand_spk,  # add sampled speaker
            temperature=0.1,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )

    def recv_data(self, data):
        self.datas = self.__spilt_text(data)
        self.__start(self.datas)
        self.datas = None


    def __start(self, datas):
        streamchat = self.chat.infer(
            datas,
            #skip_refine_text=True,
            stream=True,                        # select stream model
            # skip_refine_text=True,
            params_infer_code=self.params_infer_code,
            params_refine_text=self.params_refine_text
        )
        ChatStreamer().play(streamchat)  # now it's play audio next will fix to send audio stream

    def __spilt_text(self,data):
        lens = len(data)
        if lens > 10:
            return [data[i:i+self.step] for i in range(0, len(data), self.step)]
        else:
            return data


class TTS_Node(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        
        self.TTS = TTS()
        self.services_ = self.create_service(Int16MultiArray,"/play_audio", self.audio_stream)   # publish audio stream

        self.audio_stream_data = None
        self.subscriptions_ = self.create_subscription(String, "/speech_output", self.text_proc)      # receive text


    def audio_stream(self, msg):
        pass
        

    def text_proc(self, msg):
        data = msg.data # text
        if data is None:
            return
        self.TTS.recv_data(data)        #  return audio stream for websocket to esp32 and play      (format: wav)
        



def main(args=None):
    rclpy.init(args=args)
    node = TTS_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
