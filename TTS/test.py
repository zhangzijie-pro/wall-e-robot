import ChatTTS
from config import Config
from stream import ChatStreamer
import torch
import torchaudio

# 加载 ChatTTS

config = Config()

chat = ChatTTS.Chat()
chat.load(compile=False)


rand_spk = chat.sample_random_speaker()
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.1,  # using custom temperature
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_4]',
    #spk_emb=rand_spk,  # add sampled speaker
    temperature=0.1,  # using custom temperature
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)

spend_time = config.get_spend_time()
print(f"load_model: {spend_time} s")

# 获取ChatTTS 流式推理generator
streamchat = chat.infer(
    [
        "图像分辨率是一组用于评估图像中蕴含细节信息丰富程度的性能参数，包括时间分辨率、空间分辨率及色阶分辨率等，体现了成像系统实际所能反映物体细节信息的能力。相较于低分辨率图像，高分辨率图像通常包含更大的像素密度、更丰富的纹理细节及更高的可信赖度。但在实际上情况中，受采集设备与环境、网络传输介质与带宽、图像退化模型本身等诸多因素的约束，我们通常并不能直接得到具有边缘锐化、无成块模糊的理想高分辨率图像。提升图像分辨率的最直接的做法是对采集系统中的光学硬件进行改进，但是由于制造工艺难以大幅改进并且制造成本十分高昂，因此物理上解决图像低分辨率问题往往代价太大。由此，从软件和算法的角度着手，实现图像超分辨率重建的技术成为了图像处理和计算机视觉等多个领域的热点研究课题。"
    ],
    #skip_refine_text=True,
    stream=True,
    # skip_refine_text=True,
    params_infer_code=params_infer_code,
    params_refine_text=params_refine_text
)

ChatStreamer().play(streamchat)
