import rclpy
from rclpy.node import Node
from std_msgs.msg import Agent_msg, String
import yaml
import os
import json
import base64
from std_msgs.msg import TaskList
from split_task import generate_task_plan_from_prompt, get_multimodal_message
from rcl_interfaces.msg import SetParametersResult

ROOT_PATH = os.getcwd()
system_prompt_path = os.path.join(ROOT_PATH, "json","prompt.json")
task_template_file = os.path.join(ROOT_PATH, "json","support_task.json")

class LangchainAgentNode(Node):
    def __init__(self):
        super().__init__('cognition_layer/agent_node')

        self.system_instruction = self._load_json("system_prompt_path.json")
        self.task_template_file = self._load_json("task_template_file.json")

        self.declare_parameter("model", "qwen2.5")
        self.declare_parameter("mutil_model", "qwen2.5")

        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.mutil_model = self.get_parameter("mutil_model").get_parameter_value().string_value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.create_subscription(Agent_msg, 'cognition_layer/agent_msg', self.agent_msg_callback, 10)
        self.task_pub = self.create_publisher(String, 'cognition_layer/task_list', 10)  


    def parameter_callback(self,parameters):
        for param in parameters:
            if param.name == "model":
                self.model = param.value
            elif param.name == "mutil_model":
                self.mutil_model = param.value
        return SetParametersResult(successful = True)


    def agent_msg_callback(self, msg):
        """
        Args:
            text: String
            speaker_id: int
            img: float32[]
        """
        text_msg = msg.text
        img = msg.img
        speaker_id = msg.speaker_id
        name, history_path = self._speaker(speaker_id)
        b64_img = base64.b64encode(img).decode("utf-8")

        task_list = self._generate_task_list(text_msg, name)

        if task_list.get("vision", [False])[0]:
            multimodal_msg = get_multimodal_message(
                self.mutil_model,
                task_list["vision"]["true"],
                b64_img
            )
            task_list = self._generate_task_list(text_msg, name, extra=multimodal_msg)

        json_msg = String()
        json_msg.data = json.dumps(task_list)
        self.task_pub.publish(json_msg)
        self.get_logger().info(f"[任务生成器] 已发布任务计划：{task_list.get('goal')}")

    def _generate_task_list(self, text_msg, speaker_name, extra=None):
        return generate_task_plan_from_prompt(
            self.model,
            text_msg,
            speaker_name,
            self.system_instruction,
            self.task_template_file,
            extra=extra
        )
    
    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _speaker(self, speaker_id):
        """
        Returns speaker and his/her history
        """
        config_path = os.path.join(ROOT_PATH, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        return config['agent_node']['speaker'][speaker_id], os.path.join(ROOT_PATH, 'config', config['agent_node']['speaker_history'][speaker_id])
