import rclpy
from rclpy.node import Node
from custom_api.msg import TaskList, TaskStep

class ExecutorNode(Node):
    def __init__(self):
        super().__init__("executor_node")
        self.subscription = self.create_subscription(
            TaskList,
            "/task_list",
            self.task_list_callback,
            10
        )
        self.get_logger().info("Executor node started.")

    def task_list_callback(self, msg: TaskList):
        self.get_logger().info(f"收到任务计划：{msg.goal}")
        for i, step in enumerate(msg.steps):
            self.get_logger().info(f"步骤 {i+1}: {step.module} 执行 {step.action} 参数 {step.params}")
            self.execute_step(step)

    def execute_step(self, step: TaskStep):
        module = step.module
        action = step.action
        params = dict(step.params)  # 如果用的是 map<string, string>
        handler = f"{module}_{action}"
        if hasattr(self, handler):
            getattr(self, handler)(**params)
        else:
            self.get_logger().warn(f"未实现动作：{handler}")

    # 示例动作
    def bottom_move_to(self, location=None, target=None):
        self.get_logger().info(f"[底盘] 移动到 {location or ''} 的 {target or ''}")

    def arm_open_door(self, object=None):
        self.get_logger().info(f"[手臂] 打开 {object}")

    def vision_detect_objects(self, container=None):
        self.get_logger().info(f"[视觉] 检测 {container}")

    def speech_speak(self, message=None):
        self.get_logger().info(f"[语音] 播报：{message}")
