import rclpy
from rclpy import Node

import cv_bridge
import cv2

import sys
import os
from sensor_msg import Image


relative_path = os.getcwd()     # 获取pwd当前目录
model_path = os.path.join(relative_path, "../../","model","depth_anythingv2")

class Depth_node(Node):
    """Get Depth relative distance

    Args:
        name: node name 
    """
    def __init__(self, name="depth_node"):
        super().__init__(name)
        self.declare_parameter('type',"0")  # if 0 : double camera  else 1 : signle camera depth model

        self.declare_parameter('model', model_path)
        
    