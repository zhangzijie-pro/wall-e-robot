import random
import threading
import queue
import numpy as np
from queue_thread import SafeRealTimeBufferPlayer
from np import float_to_int16


class ChatStreamer:
    def __init__(self, base_block_size=8000):
        self.base_block_size = base_block_size

    @staticmethod
    def _update_stream(history_stream_wav, new_stream_wav, thre):
        if history_stream_wav is not None:
            result_stream = np.concatenate([history_stream_wav, new_stream_wav], axis=1)
            is_keep_next = result_stream.shape[1] < thre
        else:
            result_stream = new_stream_wav
            is_keep_next = result_stream.shape[1] < thre
        return result_stream, is_keep_next

    @staticmethod
    def _accum(accum_wavs, stream_wav):
        if accum_wavs is None:
            accum_wavs = stream_wav
        else:
            accum_wavs = np.concatenate([accum_wavs, stream_wav], axis=1)
        return accum_wavs

    @staticmethod
    def batch_stream_formatted(stream_wav, output_format="PCM16_byte"):
        if output_format in ("PCM16_byte", "PCM16"):
            return float_to_int16(stream_wav)
        return stream_wav

    @staticmethod
    def formatted(data, output_format="PCM16_byte"):
        if output_format == "PCM16_byte":
            return data.astype("<i2").tobytes()
        return data

    @staticmethod
    def checkvoice(data):
        return np.abs(data).max() >= 1e-4

    @staticmethod
    def _subgen(data, thre=1200):#2400
        for start_idx in range(0, data.shape[0], thre):
            yield data[start_idx:start_idx + thre]

    def generate(self, streamchat, output_format=None):
        assert output_format in ("PCM16_byte", "PCM16", None)
        curr_sentence_index = 0
        history_stream_wav = None
        article_streamwavs = None

        for stream_wav in streamchat:
            n_texts = len(stream_wav)
            n_valid_texts = (np.abs(stream_wav).max(axis=1) > 1e-6).sum()
            if n_valid_texts == 0:
                continue

            block_thre = n_valid_texts * self.base_block_size
            stream_wav, is_keep_next = ChatStreamer._update_stream(history_stream_wav, stream_wav, block_thre)

            if is_keep_next:
                history_stream_wav = stream_wav
                continue

            history_stream_wav = None
            stream_wav = ChatStreamer.batch_stream_formatted(stream_wav, output_format)
            article_streamwavs = ChatStreamer._accum(article_streamwavs, stream_wav)

            if ChatStreamer.checkvoice(stream_wav[curr_sentence_index]):
                for sub_wav in ChatStreamer._subgen(stream_wav[curr_sentence_index]):
                    if ChatStreamer.checkvoice(sub_wav):
                        yield ChatStreamer.formatted(sub_wav, output_format)
            elif curr_sentence_index < n_texts - 1:
                curr_sentence_index += 1
                finish_stream_wavs = article_streamwavs[curr_sentence_index]
                for sub_wav in ChatStreamer._subgen(finish_stream_wavs):
                    if ChatStreamer.checkvoice(sub_wav):
                        yield ChatStreamer.formatted(sub_wav, output_format)
            else:
                break

        if is_keep_next:
            stream_wav = ChatStreamer.batch_stream_formatted(stream_wav, output_format)
            if ChatStreamer.checkvoice(stream_wav[curr_sentence_index]):
                for sub_wav in ChatStreamer._subgen(stream_wav[curr_sentence_index]):
                    if ChatStreamer.checkvoice(sub_wav):
                        yield ChatStreamer.formatted(sub_wav, output_format)
            article_streamwavs = ChatStreamer._accum(article_streamwavs, stream_wav)

        for i_text in range(curr_sentence_index + 1, n_texts):
            finish_stream_wavs = article_streamwavs[i_text]
            for sub_wav in ChatStreamer._subgen(finish_stream_wavs):
                if ChatStreamer.checkvoice(sub_wav):
                    yield ChatStreamer.formatted(sub_wav, output_format)

    def play(self, streamchat):
        player = SafeRealTimeBufferPlayer(buffer_size=50,status_callback=lambda s: print("状态:", s))
        audio_gen = self.generate(streamchat, output_format="PCM16_byte")
        player.play(audio_gen)
