import json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_ollama import ChatOllama


import os

def generate_task_plan_from_prompt(
    model_name: str,
    user_input: str,
    user_id: str,
    system_instruction: str = "你是一个智能机器人任务拆解模块，接收自然语言并输出需要调用的子模块及其参数。",
    task_template_file: str = None,
    extra: str = None
) -> dict:
    """
    使用指定的 LLM 模型解析用户输入并输出结构化任务 JSON 计划。
    
    参数:
        model_name: 使用的 Ollama 模型名，例如 "deepseek-r1:1.5b" "qwen2.5"
        user_input: 用户的自然语言指令
        system_instruction: 系统提示词
        task_template_file: 可选，读取一个 JSON 文件作为结构体模板/上下文增强
    
    :Return:
        dict: LLM 解析出的结构化任务 JSON
    """

    llm = Ollama(model=model_name)
    output_parser = StrOutputParser()

    base_prompt = system_instruction
    if task_template_file and os.path.exists(task_template_file):
        with open(task_template_file, "r", encoding="utf-8") as f:
            template_context = json.load(f)
            base_prompt += f"\n以下是机器人系统结构与已有模块信息:\n{json.dumps(template_context, ensure_ascii=False, indent=2)}\n"
            if extra is not None:
                base_prompt += f"目前看到内容: {extra}"
    
    user_input = user_input + f"(说话人为:{user_id})"

    prompt = ChatPromptTemplate.from_messages([
        ("system", base_prompt),
        ("user", "{input}")
    ])

    chain = prompt | llm | output_parser

    response = chain.invoke({"input": user_input})

    try:
        result_json = json.loads(response)
    except json.JSONDecodeError:
        result_json = {"raw_output": response}

    return result_json


def get_multimodal_message(
        model_name: str,
        text: str,
        image_b64:str
    ):
    llm = ChatOllama(model=model_name)
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    )

    result = llm.invoke(msg)
    return result
