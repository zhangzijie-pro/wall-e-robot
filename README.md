# Wall·E Robot

<img src="./docs/images/walle.webp" alt="WALL·E" width="150"/>

Welcome to the **Wall·E Robot Project**, an intelligent, interactive robot inspired by Pixar's WALL·E. This open-source system is designed for multimodal interaction, autonomous navigation, and personalized user engagement.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<div align="center">

[中文](./docs/README_zh.md) | [English](README.md)

</div>

## 🚀 Features

* 🤖 **ESP32-Based Motion Control**

  * Controls 16-channel servo and 2 DC motors (base, arm, neck)
  * Audio signal receiver and TCP communication with Raspberry Pi

* 🧠 **Raspberry Pi + ROS2 (Humble)**

  * Voice & face recognition, TTS, STT
  * Intelligent task decomposition (LangChain + LangGraph)
  * Scene awareness and autonomous interaction

* 🔊 **Real-Time Audio Pipeline**

  * Vosk + ChatTTS with double-buffer streaming

* 🗺️ **Navigation & Mapping**

  * SLAM Toolbox, Nav2, Lidar integration (2D/3D)

* 🧩 **Modular ROS2 Nodes**

  * Full ROS2 architecture with topic/service/action interaction

---

## 🧱 Architecture

### Hardware

* **ESP32**: Motor and servo control, audio I/O
* **Raspberry Pi (Ubuntu)**: Main computation unit
* **Camera**: Face detection, object tracking
* **Lidar**: SLAM and obstacle detection

### Software Modules

* `vosk_stt_node` – Voice to text (C++)
* `chattts_node` – Text to speech (Python)
* `langchain_agent_node` – Task reasoning engine (Python)
* `executor_node` – LangGraph-based FSM scheduler (Python)
* `motion_controller_node`, `servo_controller_node` – Motor/servo control (C++)
* `slam_toolbox`, `nav2_bt_navigator` – Navigation stack

---

## 📦 Repository Structure

```bash
wall-e-robot/
│
├── 3D_Models/              # WALL·E 3D print file（e.g .stl）
├── Dataset/                # Dataset
├── depth/                  # depth anything v2 model 
├── docs/
├── ECAPA-TDNN/             # speaker recognize model 
├── ESP32_Firmware/         # ESP32 code 
├── models/                 # trained model weight
├── ROS2_Packages/          # ROS code
├── Hardware_Design/        # SCH PCB design
├── Tools/                  # download third package and model
│
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml
├── setup.py
├── LICENSE
└── README.md
```

---

## 🔧 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/zhangzijie-pro/wall-e-robot.git
```

2. **Build ROS2 Workspace**

```bash
cd wall-e-robot/ROS2_Packages
colcon build --symlink-install
source install/setup.bash
```

3. **Flash ESP32 Firmware**

* Go to `esp32_firmware/` and use PlatformIO or Arduino IDE.

4. **Launch the System**

```bash
ros2 launch launch/walle_full_system.launch.py
```

---

## 📚 Documentation

* [System Architecture (ROS2 + ESP32)](./docs/architecture.md)
* [LangGraph Task Scheduler](./docs/langgraph_fsm.md)

---

## 📝 License

Licensed under the [Apache License 2.0](LICENSE).

---

## 🤝 Contributing

Contributions are welcome via Pull Requests or ideas!  
For major changes, please open an issue first to discuss what you would like to change.  
For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🧠 Inspirations

This project is inspired by WALL·E and aims to bring emotional intelligence and autonomy to personal robots.

---

## 📬 Contact

For collaboration or questions, feel free to open an issue or reach out via GitHub.
