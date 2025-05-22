import rclpy
from rclpy.node import Node


class Camera(Node):
    def __init__(self):
        super().__init__("camera_node")

        

def main(args=None):
    rclpy.init(args=args)
    node = Camera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()