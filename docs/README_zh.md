# Wall·E 机器人

<img src="./images/walle.webp" alt="WALL·E" width="100"/>


欢迎体验 **Wall·E 机器人项目**，这是一款受皮克斯《机器人总动员》启发的智能交互机器人。该开源系统专为多模态交互、自主导航和个性化用户体验而设计。

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<div align="center">

[中文](README_zh.md) | [English](README.md)

</div>

##  🚀 功能

* 🤖 **基于 ESP32 的运动控制**

    * 控制 16 通道伺服电机和 2 个直流电机（底盘、机械臂、颈部）
    * 音频信号接收器及与 Raspberry Pi 的 TCP 通信


* 🧠 **Raspberry Pi + ROS2 (Humble)**

    * 语音和人脸识别、文本转语音（TTS）、语音转文本（STT）
    * 智能任务分解（LangChain + LangGraph）
    * 场景感知和自主交互


* 🔊 实时音频处理

    * Vosk + ChatTTS，支持双缓冲流式处理


* 🗺️ 导航与建图

    * SLAM Toolbox、Nav2、激光雷达集成（2D/3D）


* 🧩 模块化 ROS2 节点

    * 完整的 ROS2 架构，支持话题、服务和动作交互

---


## 🧱 系统架构

###  硬件

* **ESP32**：电机和伺服控制，音频输入/输出
* **Raspberry Pi (Ubuntu)**：主计算单元
* **摄像头**：人脸检测、物体跟踪
* **激光雷达**：SLAM 和障碍物检测

### 软件模块

* `vosk_stt_node` – 语音转文本（C++）
* `chattts_node` – 文本转语音（Python）
* `langchain_agent_node` – 任务推理引擎（Python）
* `executor_node` – 基于 LangGraph 的有限状态机调度器（Python）
* `motion_controller_node, servo_controller_node` – 电机/伺服控制（C++）
* `slam_toolbox, nav2_bt_navigator` – 导航堆栈

---

## 📦 仓库结构

```bash
wall-e-robot/
│
├── 3D_Models/              # WALL·E 外壳和结构的 3D 打印文件（.stl 等）
├── Dataset/                # 声纹模型、人脸识别与高清图像处理的训练数据集
├── docs
├── ESP32_Firmware/         # ESP32 上运行的嵌入式代码（例如运动控制）
├── models/                 # 训练后的模型权重
├── ROS2_Packages/          # ROS 内容
├── Hardware_Design/        # 原理图和 PCB 设计文件
├── SpeakerRecognition/     # 说话者识别模块（包含声纹识别与人脸识别）
├── tts/                    # ChatTTS 实时 TTS 推理和三缓冲播放
├── Tools/
│
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml
├── setup.py
├── LICENSE
└── README.md
```

---

## 🔧 快速入门

1. **克隆仓库**


```bash
git clone https://github.com/zhangzijie-pro/wall-e-robot.git
```

2. **构建 ROS2 工作空间**

```bash
cd wall-e-robot/ROS2_Packages
colcon build --symlink-install
source install/setup.bash
```

3. **烧录 ESP32 固件**


* 进入 esp32_firmware/，使用 vscode的ESP-IDF插件或者idf工具链进行烧录。


4. **启动系统**

```bash
ros2 launch launch/walle_full_system.launch.py
```

---

## 📚 文档

* [系统架构 (ROS2 + ESP32)](./docs/architecture.md)
* [LangGraph 任务调度器](./docs/langgraph_fsm.md)

---

## 📝 许可证
本项目采用 Apache 2.0 许可证

---

## 🤝 贡献
欢迎提交 Pull Request 或提出创意！对于重大更改，请先创建一个 Issue 进行讨论。  
详细的贡献指南请查看 [CONTRIBUTING.md](CONTRIBUTING_zh.md)

---

## 🧠 灵感来源
本项目受 WALL·E 启发，旨在为个人机器人赋予情感智能和自主能力

---

## 📬 联系方式
如需合作或有任何疑问，请通过 GitHub 创建 Issue 或直接联系
