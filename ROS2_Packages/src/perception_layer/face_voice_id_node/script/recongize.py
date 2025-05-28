import os
import numpy as np
from config import *
from log import logger,log
from embedding.embedding_predict import *
from Face.face_predict import *
from Face.face_detect import inference_img
import threading

DataSet_voice = os.path.join(PARENT_PATH,"Dataset","voice")
DataSet_face = os.path.join(PARENT_PATH,"Dataset","face_dataset")

@log()
class SpeakerRecognize_Data:
    
    image_tensor = None
    audio_tensor = None
    """
    load Data interface
    """
    def __init__(self):
        self._len = 512

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

    def check_len(self,data):
        return data[:len] if len(data)>len else data

    def __del__(self):
        """
        清除数据
        """
        self.image_tensor = None
        self.audio_tensor = None


@log()
class SpeakerRecognize:
    """
    return audio_name and [face1_name,face2_name]
    """
    def __init__(self, data:SpeakerRecognize_Data):
        self.config = Config()
        self.lib = DB_Action()
        self.device = self.config.device()

        self.__audio_data = data.audio_tensor
        self.__img_data = data.image_tensor

        self.face_detect = mnn_face_detect
        self.face_feature = mnn_face_feature
        self.voice_model = mnn_voice_feature
        
        self.lib_state = True if self.__check_path(json_lib_path_voice) is True and self.__check_path(json_lib_path_face) is True else False
        if not self.lib_state:
            self.__init_lib()
        else:
            self.__voice_db = self.lib.Get_DB(json_lib_path_voice)
            self.__face_db = self.lib.Get_DB(json_lib_path_face)

    def Start_(self):
        def run_face_process():
            try:
                self.face_result = self.__face_process()
            except Exception as e:
                self.face_result = logger.error(f"Error in face process: {e}")

        def run_voice_process():
            try:
                self.voice_result = self.__voice_process()
            except Exception as e:
                self.voice_result = logger.error(f"Error in voice process: {e}")

        face_thread = threading.Thread(target=run_face_process)
        voice_thread = threading.Thread(target=run_voice_process)

        face_thread.start()
        voice_thread.start()

        face_thread.join()
        voice_thread.join()

        return self.face_result, self.voice_result
        
    def __voice_process(self):
        (name,score) = Model_Detect_MNN(self.voice_model, self.__voice_db).identify_speaker(self.__audio_data, input_type="pcm")
        return (name,score)

    def __face_process(self):
        """
        face_detect -> face_recognize
        return :
            [{"Alice":0.7}, {"Bob":0.3}]
        """
        chain = {}
        names = []
        detect_img_tensor = inference_img(self.__img_data, self.face_detect)  # [tensor1, tensor2]
        for i in range(len(detect_img_tensor)-1):
            (name, score) = recognize_face(i,self.face_feature,self.__face_db)
            chain[name] = score
            names.append(chain)

        return names

    def __init_lib(self):
        logger.info("Init Face Audio lib now")
        audio_db = User_Voice_DB.build_voice_db(voice_path=DataSet_voice, mnn_model=self.voice_model)
        face_db = build_face_database(image_dir=DataSet_face, model_path=mnn_face_feature)
        self.lib.save(audio_db)
        self.lib.save(face_db)

    def __check_path(self, path):
        if not os.path.exists(path):
            logger.error(f"Path {path} does not exist.")
            return None
        else:
            return path