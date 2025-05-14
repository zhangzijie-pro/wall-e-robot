import os
import numpy as np
from config import *
from log import logger,log

@log()
class SpeakerRecognize_Data:
    """
    load Data interface
    """
    def __init__(self):
        self._image_tensor = None
        self._audio_tensor = None
    
    def _set_image_tensor(self, image_tensor):
        """
        this function is used to set the image tensor
        origin image -> detect -> feature
        """
        if isinstance(image_tensor, np.ndarray):
            self.image_tensor = image_tensor
        else:
            np.array(image_tensor, dtype=np.float32)
            if image_tensor is not None:
                self.image_tensor = image_tensor
            else:
                logger.error("Image tensor is None")

    def _set_audio_tensor(self, audio_tensor):
        """
        only need part of the audio
        """
        if isinstance(audio_tensor, np.ndarray):
            self.audio_tensor = audio_tensor
        else:
            np.array(audio_tensor, dtype=np.float32)
            if audio_tensor is not None:
                self.audio_tensor = audio_tensor
            else:
                logger.error("Audio tensor is None")     

    def __del__(self):
        """
        清除数据
        """
        self.image_tensor = None
        self.audio_tensor = None

@log()
class SpeakerRecognize:
    def __init__(self, data:SpeakerRecognize_Data):
        self.config = Config()
        self.device = self.config.device()
        self.data = data

        self.face_detect = mnn_face_detect
        self.face_feature = mnn_face_feature
        self.voice_model = mnn_voice_feature

        self.face_json_path = json_lib_path_face
        self.voice_json_path = json_lib_path_voice
        
        self.lib_path = [self.check_path(json_lib_path_voice), self.check_path(json_lib_path_face)]


    def check_path(self, path):
        if not os.path.exists(path):
            logger.error(f"Path {path} does not exist.")
            return None
        else:
            return path
        
    def voice_process(self):
        pass

    def face_process(self):
        pass