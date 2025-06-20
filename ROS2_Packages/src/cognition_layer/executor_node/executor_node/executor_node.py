import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_api.msg import TaskList, TaskStep
import json
import asyncio
import os
import yaml

ROOT = os.getcwd()
config_path = os.path.join(ROOT, "config", "config.yaml")

def get_config(name):
    assert not os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get(name, [])

class ExecutorNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.subscription = self.create_subscription(
            String,
            'cognition_layer/task_list',
            self.task_list_callback,
            10
        )
        for task in get_config(name):   #  {"task_name": ["msg_type","topic_name"]},
            for name, setting in task.items():
                self.pub_map[name]=self.create_publisher(setting[0], setting[1],10)

        self.get_logger().info("Executor Node Started.")

    def task_callback(self, msg: TaskList):
        step_dict = {step.id: step for step in msg.steps}

        async def dispatch_step(step: TaskStep):
            pub = self.pub_map.get(step.module)
            if pub:
                pub_msg = String()
                pub_msg.data = json.dumps({
                    "id": step.id,
                    "module": step.module,
                    "action": step.action,
                    "params": dict(step.params)
                })
                self.get_logger().info(f"Dispatching [{step.module}]: {step.action}")
                pub.publish(pub_msg)

        async def run_tasks():
            await asyncio.gather(*(dispatch_step(step_dict[i]) for i in msg.parallel))
            for i in msg.sequence:
                await dispatch_step(step_dict[i])

        asyncio.create_task(run_tasks())

def main(args=None):
    rclpy.init(args=args)
    node = ExecutorNode('executor_node')
    rclpy.spin(node)
    rclpy.shutdown()

