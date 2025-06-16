from rclpy import Node
import rclpy
from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray
from vosk import Model, KaldiRecognizer
import numpy as np
import os
import json
import asyncio
from collections import deque
import threading
import time

cwd = os.getcwd()   # Root directory of the package
config_path = os.path.join(cwd, "config", "config.yaml")

class STTNode(Node):
    def __init__(self):
        super.__init__('stt_node')
        self.declare_parameter('model_path', '../../models/vosk_model')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('buffer_size', 1024)
        self.declare_parameter('channel', 1)
        self.declare_parameter('audio_topic', 'mic_audio')
        self.declare_parameter('text_topic', 'stt_text')
        self.declare_parameter('silence_threshold', 0.1)
        self.declare_parameter('silence_duration', 0.5)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.channel = self.get_parameter('channel').get_parameter_value().integer_value
        self.audio_topic = self.get_parameter('audio_topic').get_parameter_value().string_value
        self.text_topic = self.get_parameter('text_topic').get_parameter_value().string_value
        self.silence_threshold = self.get_parameter('silence_threshold').get_parameter_value().double_value
        self.silence_duration = self.get_parameter('silence_duration').get_parameter_value().double_value

        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

        self.buffers = [deque(maxlen=self.buffer_size), deque(maxlen=self.buffer_size), deque(maxlen=self.buffer_size)]
        self.buffer_states = ['receive', 'process', 'publish']
        self.buffer_lock = asyncio.Lock()
        self.last_sound_time = time.time()
        self.current_text = ""

        self.audio_sub = self.create_subscription(
            Int16MultiArray, self.audio_topic, self.audio_callback, 10)
        self.text_pub = self.create_publisher(String, self.text_topic, 10)

        self.loop = asyncio.get_event_loop()
        self.running = True
        self.loop.create_task(self.process_audio_task())
        self.loop.create_task(self.publish_text_task())

        self.get_logger().info('STT Node initialized with async triple buffering')

    def audio_callback(self, msg):
        if len(msg.data) != self.buffer_size * self.channel:
            self.get_logger().warn(f"Received audio array size {len(msg.data)}, expected {self.buffer_size}")
            return
        asyncio.run_coroutine_threadsafe(self.add_to_receive_buffer(msg.data), self.loop)

    async def add_to_receive_buffer(self, audio_data):
        async with self.buffer_lock:
            receive_buffer = next(b for b, state in zip(self.buffers, self.buffer_states) if state == 'receive')
            receive_buffer.clear()
            receive_buffer.extend(audio_data)
            self.swap_buffers()

    def swap_buffers(self):
        self.buffer_states = self.buffer_states[1:] + [self.buffer_states[0]]

    async def process_audio_task(self):
        while self.running:
            async with self.buffer_lock:
                process_buffer = next((b for b, state in zip(self.buffers, self.buffer_states) if state == 'process'), None)
                if process_buffer and len(process_buffer) == self.buffer_size:
                    audio_data = np.array(list(process_buffer), dtype=np.int16)
                    audio_bytes = audio_data.tobytes()

                    energy = np.abs(audio_data).mean()
                    is_speech = energy > self.silence_threshold * 32768

                    if is_speech:
                        self.last_sound_time = time.time()

                    if self.recognizer.AcceptWaveform(audio_bytes):
                        result = json.loads(self.recognizer.Result())
                        if 'text' in result and result['text']:
                            self.current_text = result['text']
                            self.swap_buffers()
                    else:
                        partial_result = json.loads(self.recognizer.PartialResult())
                        if 'partial' in partial_result and partial_result['partial']:
                            self.get_logger().debug(f"Partial: {partial_result['partial']}")

                    if not is_speech and time.time() - self.last_sound_time > self.silence_duration:
                        final_result = json.loads(self.recognizer.FinalResult())
                        if 'text' in final_result and final_result['text']:
                            self.current_text = final_result['text']
                        self.recognizer.Reset()
                        self.swap_buffers()

            await asyncio.sleep(0.01)

    async def publish_text_task(self):
        while self.running:
            if self.current_text:
                async with self.buffer_lock:
                    publish_buffer = next((b for b, state in zip(self.buffers, self.buffer_states) if state == 'publish'), None)
                    if publish_buffer:
                        msg = String()
                        msg.data = self.current_text
                        self.text_pub.publish(msg)
                        self.get_logger().info(f"Published text: {self.current_text}")
                        self.current_text = ""
                        self.swap_buffers()
            await asyncio.sleep(0.05)
    
    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()