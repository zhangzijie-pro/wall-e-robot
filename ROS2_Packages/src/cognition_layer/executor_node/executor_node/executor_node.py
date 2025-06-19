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
        self.get_logger().info(f"✅ This Task Goal: {msg.goal}")
        for step in msg.steps:
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

