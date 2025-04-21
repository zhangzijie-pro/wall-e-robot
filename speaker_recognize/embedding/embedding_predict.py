import torch
import torchaudio
import librosa
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from embedding import TrckNet
import os
from embedding import logger

class User_DB:
    def __init__(self):
        self.db = {}
        self.embs = []

    def extract_mel(self, file_path, sr=16000, n_mels=128, n_fft=512, hop_length=160, win_length=400):
        y, _ = librosa.load(file_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                             n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        log_mel = librosa.power_to_db(mel)
        return log_mel

    def preprocess(self,file_path):
        mel = self.extract_mel(file_path)  # [n_mels, time]
        mel = torch.tensor(mel).unsqueeze(0)  # [1, n_mels, T]
        mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(128, 128), mode='bilinear')  # [1, 1, 128, 128]
        return mel.squeeze(0)

    def build_voice_db(self, voice_path, model):
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
        """
        for speaker in os.listdir(voice_path):
            speaker_dir = os.path.join(voice_path, speaker)
            if not os.path.isdir(speaker_dir): continue
            for file in os.listdir(speaker_dir):
                if file.endswith(".wav"):
                    audio_path = os.path.join(speaker_dir, file)
                    mel = self.preprocess(audio_path)
                    with torch.no_grad():
                        emb = model(mel.unsqueeze(0))
                        self.embs.append(emb.squeeze(0))
            if self.embs:
                speaker_embedding = torch.stack(self.embs).mean(dim=0)
                self.db[speaker] = speaker_embedding
        return self.db
    

class DB_Action:
    """
    DB local save
    Args:
        save_format: "pickle" / "json" / "redis"
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
            data: dict
            save_path: you can input path or file_path (json/pkl)
            Example: "./" or "./data.pkl"

        Example save to pkl:
        .. code-block:: python
            data = {"name":"-0.231,0.310004,-0.14"}
            save_path = "./"
            DB_Action(save_format="pkl").save(data, save_path)
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

    def __save_pickle(self, data):
        """Save data as a pickle file."""
        import pickle
        data_serializable = {label: embedding.tolist() for label, embedding in data.items()}
        with open(self.__save_path, "wb") as file:
            pickle.dump(data_serializable, file)

    def __save_json(self,data):
        """Save data as a JSON file."""
        import json
        data_serializable = {label: embedding.tolist() for label, embedding in data.items()}
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
            logger.error()
            raise RuntimeError(f"Failed to save to Redis: {e}")
        finally:
            self.r.close()

    def Get_DB(self) -> dict:
        """
        Get Json Pkl Redis  K-V 

        Example save to pkl:
        .. code-block:: python
            data = {}
            db_act = DB_Action(save_format="pkl")
            db_act.save(data, save_path)
            data = db_act.Get_DB()
        """
        assert not os.path.exists(self.__save_path), f"Can't found File to Read"
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
    Predict speaker

    Args:
        model: TrckNet
        model_path: PTH file address
        db: User voiceprint registry (dict)
        threshold: Cosine similarity threshold
    ⚠️ The threshold must be less than 1

    """
    def __init__(self, model:TrckNet, model_path, db={}, threshold=0.8):
        self.model_path = model_path
        assert not os.path.exists(model_path), f"Model Path {model_path} not exist"
        self.db = db
        self.threshold= threshold
        assert threshold<1, f"threshold must <=1"
        self.model = model
        model.load_state_dict(torch.load(self.model_path, map_location='cpu')) 

    def identify_speaker(self, wav_path=""):
        assert (wav_path.endswith(".wav") and os.path.exists(wav_path)), f"{wav_path} format error or don't exist"
        mel = User_DB().preprocess(wav_path)
        with torch.no_grad():
            emb = self.model(mel.unsqueeze(0))  # [1, 128] Get wav feature
        score = []
        name = []
        for label, feature_vector in self.db.items():
            sim = self.__cosine_similarity(a=emb,b=feature_vector)
            if sim>self.threshold:
                score.append(sim) 
                name.append(label)
                logger.info(f"✅ sound like is {name}, score get {sim:.4f}")
        
        return name[self.__get_max(score)] 
 
    def __get_max(self, data):
        """
        Get max score in data 
        return max idx
        """
        if not data:
            logger.warning("⚠️ data is empty")
        return max(enumerate(data), key=lambda x: x[1])[0]

    def __cosine_similarity(self,a, b):
        return F.cosine_similarity(a, b).item()
