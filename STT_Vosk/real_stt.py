"""
vosk: rate: 16khz   channel: 1  
"""
from functools import wraps
import queue
import threading
import time
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import psutil
from SpeakerRecognition.log import log,logger

@log()
class VoiceProcessor:
    def __init__(self):
        self.last_voice_time = time.time()
        self.model = Model("wall-e-robot\STT\stt")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio_queue = queue.Queue()

    def audio_callback(self, indata, frames, timestamp, status):
        self.audio_queue.put(bytes(indata))
        self.last_voice_time = time.time()  

    def record_thread(self):
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.audio_callback
        ):
            print("录音设备已启动")
            while True:
                time.sleep(0.1)

    def recognition_thread(self):
        while True:
            data = self.audio_queue.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                print(f"\n最终识别：{result['text']}")
            else:
                partial = json.loads(self.recognizer.PartialResult())
                print(f"\r部分结果：{partial.get('partial','')}", end="", flush=True)


if __name__ == "__main__":
    processor = VoiceProcessor()
    
    threading.Thread(target=processor.record_thread, daemon=True).start()
    threading.Thread(target=processor.recognition_thread, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务已安全停止")
