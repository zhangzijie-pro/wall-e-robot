from .embedding.embedding import TrckNet, model_train, Config, voice_path
from .embedding.embedding_predict import User_Voice_DB, DB_Action,Model_Detect
import os

# pkl  json  redis
os.environ["Save_format"] = "pkl"
os.environ["speaker_model_path"] = r"../model/tvector_model.pth"

speaker_model = TrckNet()


def Save_Voice_db(save_format: str = "pkl"):
    """
    存储现有数据集声纹数据内容
    """
    format =  save_format if not os.getenv("Save_format") else os.getenv("Save_format")

    db = User_Voice_DB().build_voice_db(voice_path, speaker_model)

    DB_Action(save_format=format).save(db)


def additional_user(data,key="user0",save_format: str = "pkl",input_type: str = "file"):
    """
    添加新增声纹向量信息
    """
    format =  save_format if not os.getenv("Save_format") else os.getenv("Save_format")
    vector = User_Voice_DB().preprocess(data,input_type)

    db = {key:vector}

    DB_Action(save_format=format).save(db)


def Model_Train():
    """Train model"""
    device = Config.device()
    model_train(device=device)    # 训练模型

def Detect(Detect: Model_Detect, data):
    """
    声纹识别预测
    """
    result = Detect.identify_speaker(data)

    return result
