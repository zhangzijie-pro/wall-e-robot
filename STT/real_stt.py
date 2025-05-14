"""
vosk: rate: 16khz   channel: 1  
"""
from functools import wraps
import queue
import threading
import time  # 新增导入
import json  # 新增导入（识别结果需要）
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import psutil
from speaker_recognize.log import logger, log


@log()
class VoiceProcessor:
    def __init__(self):
        self.last_voice_time = time.time()  # 使用系统时间初始化
        self.model = Model("wall-e-robot\STT\stt")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio_queue = queue.Queue()

    # 修改后的回调函数
    def audio_callback(self, indata, frames, timestamp, status):
        """音频输入回调函数 (参数名timestamp避免冲突)"""
        self.audio_queue.put(bytes(indata))
        # 正确使用系统时间（time模块）
        self.last_voice_time = time.time()  
        # 如果需要专业音频时间戳，可访问：
        #print(f"输入缓冲时间: {timestamp.inputBufferAdcTime}")

    def record_thread(self):
        """录音线程"""
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.audio_callback  # 绑定类方法作为回调
        ):
            print("录音设备已启动")
            while True:
                time.sleep(0.1)  # 防止CPU占用过高

    def recognition_thread(self):
        """识别线程"""
        print("识别引擎就绪")
        while True:
            data = self.audio_queue.get()
            # print("data:", data, len(data))
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                print(f"\n最终识别：{result['text']}")
            else:
                partial = json.loads(self.recognizer.PartialResult())
                print(f"\r部分结果：{partial.get('partial','')}", end="", flush=True)


if __name__ == "__main__":
    processor = VoiceProcessor()
    
    # 创建并启动线程
    threading.Thread(target=processor.record_thread, daemon=True).start()
    threading.Thread(target=processor.recognition_thread, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务已安全停止")
