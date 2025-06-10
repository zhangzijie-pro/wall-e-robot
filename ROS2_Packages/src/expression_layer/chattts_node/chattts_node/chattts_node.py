import rclpy
from rclpy import Node
import rclpy.node
from std_msgs.msg import String
import ChatTTS
from stream import ChatStreamer

import asyncio
import websockets
import threading
import base64
import json

class TTS:
    """
    ChatTTS
    """
    def __init__(self):
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)

        self.datas = None
        self.step = 8
        

        self.turn_flag = False
        self.audio_stack = []
        self.recv_state = False

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

    def close_recv_state(self):
        self.recv_state = False

    def recv_callback(self):
        self.recv_state = True

    def pop_audio_chunk(self):
        if self.audio_stack:
            return self.audio_stack.pop(0)
        return None

    def __infer(self, data, status):
        self.datas = self._split_text(data)
        for data in self.datas:
            streamchat = self.chat.infer(
                data,
                #skip_refine_text=True,
                stream=True,                        # select stream model
                # skip_refine_text=True,
                params_infer_code=self.params_infer_code,
                params_refine_text=self.params_refine_text
            )
            audio_chunk = ChatStreamer().audio_part(streamchat,status_callback=status)
            self.audio_stack.append(audio_chunk)   # now it's play audio next will fix to send audio stream
            self.turn_flag = True if len(self.audio_stack)>=2 else False
        
        self.datas = None

    def _split_text(self, text):
        if len(text) > 10:
            return [text[i:i+self.step] for i in range(0, len(text), self.step)]
        return [text]


class TTS_Node(Node):
    """
    TTS node
    """
    def __init__(self):
        super().__init__("expression_layer/tts_node")
        
        self.TTS = TTS()
        self.subscriptions_ = self.create_subscription(
            String, 
            "/speech_output", 
            self.text_proc,
            10
            )      # receive text

    def text_callback(self, msg):
        text = msg.data
        if not text:
            return

        threading.Thread(target=self._process_and_send_audio, args=(text,), daemon=True).start()

    def _process_and_send_audio(self, text):
        self.TTS.__infer(text, status=lambda s: self.get_logger.info("state:", s))
        while True:
            audio_chunk = self.TTS.pop_audio_chunk()
            if audio_chunk is None:
                break
            
            encoded = base64.b64encode(audio_chunk[0]).decode("utf-8")

            data = {
                "type": "audio_chunk",
                "payload": encoded,
                "format": "PCM16"
            }

            asyncio.run(self._send_ws(data))

    async def _send_ws(self, data):
        uri = "ws://127.0.0.1/audio_part"
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(data))
        except Exception as e:
            self.get_logger().error(f"WebSocket Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TTS_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
