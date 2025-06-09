import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

def depth_np(file_name, raw_img=None, input_size=518, save_type="numpy"):
    """
    
    """
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}

    depth_anything = DepthAnythingV2(**model_configs["vits"])
    depth_anything.load_state_dict(torch.load(r'./depth_anything_v2_vits.pth', map_location='cpu',weights_only=False))
    depth_anything = depth_anything.to(DEVICE).eval()

    depth = depth_anything.infer_image(raw_image=raw_img,input_size=input_size)   # img: HWC  -> float 32

    if save_type=="png" or save_type=="jpg":
        depth = (depth-depth.min()) / (depth.max()-depth.min()) * 255.0  # 计算得出相对深度占比  归一化
        depth = depth.astype(np.uint8)      # HWC -> HW

        # TO Gray
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # HW -> HWC 
        cv2.imwrite(os.path.join("./output/", file_name + '.png', depth))

    elif save_type=="numpy":
        np.save('depth2.npy', depth)
    else:
        # logger.error(f"Can't support {save_type}")
        pass

    return depth