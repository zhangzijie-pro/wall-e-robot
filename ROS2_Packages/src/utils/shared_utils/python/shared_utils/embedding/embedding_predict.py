import torch
import torchaudio
import librosa
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from embedding import TrckNet,voice_path
import os
from embedding import logger
import base64
import io
import soundfile as sf
import MNN

class User_Voice_DB:
    """
    Build User Voice DB
    there will get voice mel feature and user db
    """
    def __init__(self):
        self.db = {}
        self.embs = []

    def extract_mel(self, data, sr=16000, n_mels=128, n_fft=512, hop_length=160, win_length=400):
        # y, _ = librosa.load(data, sr=sr)
        mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels,
                                             n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        log_mel = librosa.power_to_db(mel)
        return log_mel

    def _load_audio(self,data, sr=16000, input_type="file"):
        if input_type == "file":
            y, _ = librosa.load(data,sr=sr)
        elif input_type == 'stream':
            y = data.astype(np.float32)
        elif input_type == "base64":
            audio_bytes = base64.b64decode(data)
            audio_np, _ = sf.read(io.BytesIO(audio_bytes),dtype="float32")
            y = audio_np
        elif input_type == "pcm":
            y = data.astype(np.float32)
        else:
            logger.error("%s not Support to know"%input_type)
        
        return y
        

    def preprocess(self,data,input_type="file"):
        """
        input_type
        - 'file'：路径字符串
        - 'stream': numpy array(如Vosk队列中获取的)
        - 'base64': base64编码字符串
        - 'pcm': ESP32发送的PCM numpy数组

        Example file data (wav)
        .. code-block:: python
            data = r"voice_path"
            mel = preprocess(data, input_type="file")
            
        Example stream data 
        .. code-block:: python
            mel = preprocess(stream_data, input_type="steam")

        Example base64 soundfile
        .. code-block:: python
            with open("sample.wav", "rb") as f:
                b64_audio = base64.b64encode(f.read()).decode("utf-8")

            mel = preprocess(b64_audio, input_type="base64")
          
        Example pcm data (hardware i2s)
        .. code-block:: python
            mel = preprocess(i2s_pcm_array, input_type="pcm")

        Returns:
            mel feature
        """
        y = self._load_audio(data, input_type=input_type)
        mel = self.extract_mel(data=y)  # [n_mels, time]
        mel = torch.tensor(mel).unsqueeze(0)  # [1, n_mels, T]
        mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(128, 128), mode='bilinear')  # [1, 1, 128, 128]
        return mel.squeeze(0)


    def build_voice_db(self, voice_path:str, mnn_model:str):
        """
        Get DB -> {speaker: feature vector, ...}
        you can save this labels and features to you file

        Args:
            voice_path:voice dataset path
            model: TrckNet or other speaker model

        Example save to pickle
        .. code-block:: python
            import pickle
            
            data = User_DB().build_voice_db(voice_path, model)
            with open('data.pkl', 'wb') as file:
                pickle.dump(data, file)
        
        Example save to json
        .. code-block:: python
            import json
            
            data = User_DB().build_voice_db(voice_path, model)
            with open('data.json', 'wb') as file:
                json.dump(data, file) 

        Returns:
            db(dict)     
        """
        # use PT model build
        # for speaker in os.listdir(voice_path):
        #     speaker_dir = os.path.join(voice_path, speaker)
        #     if not os.path.isdir(speaker_dir): continue

        #     embeddings = []
        #     for file in os.listdir(speaker_dir):
        #         if file.endswith(".wav"):
        #             audio_path = os.path.join(speaker_dir, file)
        #             mel = self.preprocess(audio_path)
        #             with torch.no_grad():
        #                 emb = model(mel.unsqueeze(0))  # [1, 128]
        #                 embeddings.append(emb.squeeze(0))  # [128]

        #     if embeddings:
        #         self.db[speaker] = embeddings  # List[Tensor]
        # return self.db
        import numpy as np

        for speaker in os.listdir(voice_path):
            speaker_dir = os.path.join(voice_path, speaker)
            if not os.path.isdir(speaker_dir):
                continue

            embeddings = []
            for file in os.listdir(speaker_dir):
                if file.endswith(".wav"):
                    audio_path = os.path.join(speaker_dir, file)
                    
                    # 预处理获取梅尔频谱（假设返回PyTorch Tensor）
                    mel_tensor = self.preprocess(audio_path)
                    
                    # mel_np = mel_tensor.numpy().astype(np.float32)
                    # mel_np = np.expand_dims(mel_np, axis=0)  # 添加批次维度 [1, ...]
                    
                    emb_tensor = __infer(mel_tensor, mnn_model)
                    
                    embeddings.append(emb_tensor)

            if embeddings:
                self.db[speaker] = embeddings
        return self.db
    
    
    def __data_db__(self):
        return self.db

def __infer(mel,model_path):
    """
    Use MNN model to infer embedding
    """
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    mel = mel.unsqueeze(0)  # [1, dim, time]
    mel = mel.numpy()

    # Prepare input
    mel = np.expand_dims(mel, 0) if mel.ndim == 3 else mel  # (batch, channel, dim, time)
    tmp_input = MNN.Tensor(input_tensor.getShape(), MNN.Halide_Type_Float, mel, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    
    interpreter.runSession(session)
    
    output = interpreter.getSessionOutput(session)
    output_np = np.array(output.getData(), dtype=np.float32)

    return torch.from_numpy(output_np)  # 转回 torch tensor，方便后续余弦相似度计算

class DB_Action:
    """
    DB local save and Get
    
    Args:
        save_format: "pickle", "json", "redis"
        "redis": redis-host, port, <password>
    """
    def __init__(self,save_format="pkl",host='localhost', port=6379, password=""):
        self.save_format = save_format
        self.host = host
        self.port = port
        self.password = password
        self.__save_path = None
        self.db = {}
        self.r = None
    
    def save(self, data, save_path="./"):
        """
        support Json Pkl Redis
        
        Args:
            - data: dict
            - save_path: you can input path or file_path (json/pkl)
            - Example: "./" or "./data.pkl"

        Example save to pkl:

        .. code-block:: python
            data = {"name":"-0.231,0.310004,-0.14"}
            save_path = "./"
            DB_Action(save_format="pkl").save(data, save_path)
        
        Returns:
            None
        """
        import os
        if save_path.endswith(".pkl") or save_path.endswith(".json"):
            self.__save_path=save_path
        else:
            assert os.path.isdir(save_path), f"Directory {save_path} does not exist or This is not Dir"
            self.__save_path = os.path.join(save_path, f"data.{self.save_format}")


        if self.save_format=="pkl":
            self.__save_pickle(data)
        elif self.save_format=="json":
            self.__save_json(data)
        elif self.save_format=="reids":
            self.__save_redis(data)
        else:
            logger.error("Unsupported save format:%s"%self.save_format)
            raise ValueError(f"Unsupported save format: {self.save_format}")
        logger.info(f"Save {self.save_format} to {self.__save_path}")

    def __save_pickle(self, data):
        """Save data as a pickle file."""
        import pickle
        data_serializable = {
            label: [tensor.tolist() for tensor in embeddings]
            for label, embeddings in data.items()
        }
        with open(self.__save_path, "wb") as file:
            pickle.dump(data_serializable, file)

    def __save_json(self,data):
        """Save data as a JSON file."""
        import json
        data_serializable = {
            label: [tensor.tolist() for tensor in embeddings]
            for label, embeddings in data.items()
        }
        with open(self.__save_path, "w") as file:
            json.dump(data_serializable, file, indent=2)
    
    def __save_redis(self, data):
        import redis,json
        try:
            self.r = redis.Redis(
                host=self.host,
                port=self.port,
                db=0,
                password=self.password,
                decode_responses=True
            )
            for label, embedding in data.items():
                key = label
                emb_str = json.dumps(embedding.tolist())
                self.r.set(key, emb_str)
        except redis.RedisError as e:
            logger.error(e)
            # raise RuntimeError(f"Failed to save to Redis: {e}")
        finally:
            self.r.close()

    def Get_DB(self, data_path=None) -> dict:
        """
        Get Json Pkl Redis  K-V 

        Example save to pkl:
        .. code-block:: python
            data = {}
            db_act = DB_Action(save_format="pkl")
            db_act.save(data, save_path)
            data = db_act.Get_DB()

        Returns:
            db(dict)
        """
        # assert not os.path.exists(self.__save_path), f"Can't found File to Read"
        if not data_path is None:
            self.__save_path = data_path

        if self.save_format=="pkl":
            self.__Get_pkl()
            return self.db
        elif self.save_format=="json":
            self.__Get_json()
            return self.db
        elif self.save_format=="reids":
            self.__Get_redis()
            return self.db
        else:
            logger.error("Unsupported save format: %s"%self.save_format)
            raise ValueError(f"Unsupported save format: {self.save_format}")
        
    def __Get_json(self):
        import json
        with open(self.__save_path, "r") as f:
            self.db = json.load(f)

    def __Get_pkl(self):
        import pickle
        with open(self.__save_path, "rb") as f:
            self.db = pickle.load(f)
    
    def __Get_redis(self):
        import json,redis
        for key in self.r.scan_iter(match="*"):
            label = key.decode().split(":")[1]
            emb_list = json.loads(self.r.get(key).decode())
            self.db[label] = torch.tensor(emb_list)

class Model_Detect:
    """
    Speaker recognition engine
    """

    def __init__(self, model, model_path, db={}, threshold=0.8):
        assert threshold < 1, f"threshold must < 1"
        self.model_path = model_path
        self.model = model
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        self.db = db
        self.threshold = threshold
        print(f"✅ Loaded model from {model_path}")

    def identify_speaker(self, data, type_file="file"):
        mel = User_Voice_DB().preprocess(data, input_type=type_file)

        with torch.no_grad():
            emb = self.model(mel.unsqueeze(0)).squeeze(0)  # [128]

        names = []
        scores = []

        for label, embeddings in self.db.items():
            for ref_emb in embeddings:
                sim = self.__cosine_similarity(emb, ref_emb)
                print(f"✅ Match {label}, similarity: {sim:.4f}")
                # if sim > best_score:
                    # best_score = sim
                    # best_label = label
                scores.append(sim)
                names.append(label)

        # if best_score < self.threshold:
        #     print(f"⚠️ No match found, highest score {best_score:.4f}")
        #     return "Unknown"
        # else:
        #     return best_label
        return names[self.__get_max(scores)]
 
    def __get_max(self, data):
        """
        Get max score in data 
        return max idx
        """
        max_score=0
        max_idx=0
        if not data:
            logger.warning("⚠️ data is empty")
        for idx, score in enumerate(data):
            if score>max_score:
                max_score = score
                max_idx = idx   
        return max_idx

    def __cosine_similarity(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


import MNN
import numpy as np
import os
import torch
import torch.nn.functional as F

class Model_Detect_MNN:
    """
    Predict speaker (MNN version)

    Args:
        model_path: MNN file address
        db: User voiceprint registry (dict)
        threshold: Cosine similarity threshold
    ⚠️ The threshold must be less than 1
    """
    
    def __init__(self, model_path, db={}, threshold=0.8):
        assert os.path.exists(model_path), f"Model Path {model_path} not exist"
        self.model_path = model_path
        self.db = db
        self.threshold = threshold
        assert threshold < 1, "threshold must be less than 1"
        
        # Load MNN model
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

    def identify_speaker(self, data, input_type="file"):
        """
        identify speaker

        Args:
            data: input data example file or stream data
            input_type:
                - 'file'：路径字符串
                - 'stream': numpy array(如Vosk队列中获取的)
                - 'base64': base64编码字符串
                - 'pcm': ESP32发送的PCM numpy数组
        
        Returns:
            speaker name or label
        """
        mel = User_Voice_DB().preprocess(data, input_type=input_type)  # [dim, time]
        
        # Step 2: MNN 推理
        emb = self.__infer(mel)  # emb shape: [128]
        
        # Step 3: 比对声纹
        score = []
        name = []
        for label, feature_vector in self.db.items():
            sim = self.__cosine_similarity(a=emb, b=feature_vector)
            if sim > self.threshold:
                score.append(sim)
                name.append(label)
                logger.info(f"✅ sound like is {name}, score get {sim:.4f}")
        
        if not score:
            logger.warning("⚠️ No matching speaker found")
            return None
        
        return name[self.__get_max(score)] 

    def __infer(self, mel):
        """
        Use MNN model to infer embedding
        """
        mel = mel.unsqueeze(0)  # [1, dim, time]
        mel = mel.numpy()

        # Prepare input
        mel = np.expand_dims(mel, 0) if mel.ndim == 3 else mel  # (batch, channel, dim, time)
        tmp_input = MNN.Tensor(self.input_tensor.getShape(), MNN.Halide_Type_Float, mel, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        
        self.interpreter.runSession(self.session)
        
        output = self.interpreter.getSessionOutput(self.session)
        output_np = np.array(output.getData(), dtype=np.float32)
        return torch.from_numpy(output_np)  # 转回 torch tensor，方便后续余弦相似度计算

    def __get_max(self, data):
        """
        Get max score in data 
        return max idx
        """
        if not data:
            logger.warning("⚠️ data is empty")
            return 0
        return max(enumerate(data), key=lambda x: x[1])[0]

    def __cosine_similarity(self, a, b):
        return F.cosine_similarity(a, b, dim=0).item()



current_dir = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(current_dir ,'..','..', 'model','tvector_model.pth'))
model = TrckNet()
db = User_Voice_DB().build_voice_db(voice_path=voice_path, model=model)
DB_Action(save_format="json").save(db)
x2_path = r"C:\Users\lenovo\Desktop\python\音频信号处理\1.wav"
detect = Model_Detect(model= model,model_path=model_path, db=db)
x = detect.identify_speaker(x2_path)
print(x)
 