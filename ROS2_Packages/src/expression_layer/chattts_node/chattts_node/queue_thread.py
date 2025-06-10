import threading
import time
import pyaudio
from collections import deque
import queue


class SafeRealTimeBufferPlayer:
    def __init__(self, rate=24000, channels=1, chunk=1024, buffer_size=50, status_callback=None):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.buffer_size = buffer_size

        self.buffers = [deque(maxlen=buffer_size) for _ in range(3)]
        self.buffer_index = 0


        self.lock = threading.Lock()

        self.finished = threading.Event()
        self.playing = threading.Event()
        self.paused = threading.Event()

        self.audio_queue = queue.Queue(maxsize=buffer_size * 3)
        self.status_callback = status_callback

        self.audio_gen_iter = None
        self.producer_thread_started = False


        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.channels,
                                  rate=self.rate,
                                  output=True)
    def _get_buffers(self):
        with self.lock:
            return (self.buffers[self.buffer_index%3],
                    self.buffers[(self.buffer_index+1)%3],
                    self.buffers[(self.buffer_index+3)%3]
                    )
    
    def _advance_buffers(self):
        with self.lock:
            self.buffer_index = (self.buffer_index+1)%3

    # def _swap_buffers(self):
    #     with self.lock:
    #         self.current_buffer, self.next_buffer = self.next_buffer, self.current_buffer


    def _fill_buffer(self, buf: deque):
        try:
            while len(buf) < self.buffer_size:
                data = self.audio_queue.get(timeout=0.25)
                buf.append(data)
        except queue.Empty:
            pass

    def _producer(self, audio_gen):
        for data in audio_gen:
            while self.audio_queue.full():
                time.sleep(0.01)
            self.audio_queue.put(data)
        self.finished.set()

    def _invoke_callback(self):
        if self.status_callback:
            status = self.get_status()
            self.status_callback(status)

    def get_status(self):
        current, next_b, prefetch = self._get_buffers()
        return {
            "playing": self.playing.is_set(),
            "paused": self.paused.is_set(),
            "buffer_current": len(current),
            "buffer_next": len(next_b),
            "buffer_prefetch": len(prefetch),
            "queue_size": self.audio_queue.qsize(),
            "finished": self.finished.is_set()
        }

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def stop(self):
        self.playing.clear()


    def play(self, audio_gen):
        producer_thread = threading.Thread(target=self._producer, args=(audio_gen,))
        producer_thread.daemon = True
        producer_thread.start()

        self.playing.set()
        self.paused.clear()

        # 初始填充 next 和 prefetch
        _, next_b, prefetch = self._get_buffers()
        self._fill_buffer(next_b)
        self._fill_buffer(prefetch)

        while self.playing.is_set():
            if self.paused.is_set():
                time.sleep(0.1)
                continue

            current, next_b, prefetch = self._get_buffers()

            if len(next_b) < self.buffer_size:
                # 如果下一块未准备好就等待
                self._fill_buffer(next_b)
                continue

            # 进入播放 current（也就是 current = next_b）
            self._advance_buffers()

            while current and self.playing.is_set() and not self.paused.is_set():
                data = current.popleft()
                self.stream.write(data)
                self._invoke_callback()

            # 播放过程中填充新的 prefetch
            _, _, prefetch = self._get_buffers()
            threading.Thread(target=self._fill_buffer, args=(prefetch,), daemon=True).start()

            if self.finished.is_set() and not current and not next_b:
                break

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    
    def prepare(self, audio_gen):
        self.audio_gen_iter = iter(audio_gen)
        self.playing.set()
        self.paused.clear()

        if not self.producer_thread_started:
            threading.Thread(target=self._producer, daemon=True).start()
            self.producer_thread_started = True

        _, next_b, prefetch = self._get_buffers()
        self._fill_buffer(next_b)
        self._fill_buffer(prefetch)

    def get_next_audio_chunk(self):
        if not self.playing.is_set():
            return None
        
        if self.paused.is_set():
            return None
        
        current, next_b, prefetch = self._get_buffers()

        if len(next_b) < self.buffer_size:
            self._fill_buffer(next_b)
            return None
        
        if not current:
            self._advance_buffers()
            current, next_b, prefetch = self._get_buffers()
            threading.Thread(target=self._fill_buffer, args=(prefetch,), daemon=True).start()

        if current:
            data = current.popleft()
            self._invoke_callback()
            return data
        elif self.finished.is_set() and not next_b:
            self.playing.clear()
            return None
        return None