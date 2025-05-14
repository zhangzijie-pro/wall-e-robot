# 贡献 Wall·E 机器人项目

感谢您对 Wall·E 机器人项目的兴趣！我们欢迎社区的贡献，共同改进这个开源的智能交互机器人系统。本文档概述了贡献指南，以确保协作过程顺利进行。

---

## 如何贡献

贡献可以包括代码、文档、错误报告、功能请求或改进建议。以下是开始的步骤。

### 1. 报告问题
- **检查现有问题**：在创建新问题前，请搜索 [GitHub Issues](https://github.com/zhangzijie-pro/wall-e-robot/issues) 以避免重复。
- **创建新问题**：
  - 使用清晰且描述性的标题。
  - 提供详细信息，包括：
    - 重现问题的步骤。
    - 预期和实际行为。
    - 相关日志、截图或硬件/软件细节。
  - 如果有问题模板（如错误报告、功能请求），请使用相应模板。

### 2. 提交拉取请求（PR）
- **Fork 仓库**：
  - 将 [Wall·E 机器人仓库](https://github.com/zhangzijie-pro/wall-e-robot) Fork 到您的 GitHub 账户。
  - 本地克隆您的 Fork：
    ```bash
    git clone https://github.com/YOUR_USERNAME/wall-e-robot.git
    ```
- **创建分支**：
  - 为您的更改创建一个新分支：
    ```bash
    git checkout -b feature/your-feature-name
    ```
  - 使用描述性分支名称（例如 `fix/audio-bug`、`docs/update-readme`）。
- **进行更改**：
  - 遵循项目使用的编码风格和规范（见 [编码风格](#编码风格)）。
  - 确保更改经过充分测试并附带文档。
  - 如有必要，更新相关文档（例如 README、architecture.md）。
- **提交更改**：
  - 编写清晰、简洁的提交信息，遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式，例如：
    ```
    feat: 添加语音识别超时处理
    fix: 修复伺服控制器内存泄漏
    docs: 更新架构图
    ```
- **推送并创建拉取请求**：
  - 将分支推送到您的 Fork：
    ```bash
    git push origin feature/your-feature-name
    ```
  - 针对主仓库的 `main` 分支打开拉取请求。
  - 使用 PR 模板（如果有）并提供：
    - 更改的清晰描述。
    - 相关问题的引用（例如 `Fixes #123`）。
    - 任何额外的上下文或测试说明。
- **代码审查**：
  - 回复维护者的反馈并进行必要更改。
  - 确保所有自动化检查（例如 CI 测试、代码格式检查）通过。

### 3. 提出功能或创意
- 在 [GitHub Discussions](https://github.com/zhangzijie-pro/wall-e-robot/discussions) 或 Issue 中分享您的想法。
- 清楚说明功能、其优势及可能的实现细节。
- 与社区互动以完善提案，然后再开始开发。

---

## 编码风格

为保持代码库的一致性，请遵循以下指南：

### 通用
- **语言**：代码、注释、文档和提交信息使用英语。
- **格式**：遵循相应语言或框架的格式规范。
- **文档**：为新增或修改的代码添加或更新注释和文档。

### C++（ESP32 和 ROS2 节点）
- 遵循 [ROS2 C++ 风格指南](https://docs.ros.org/en/humble/Contributing/Code-Style-Guidelines.html#cpp)。
- 变量名使用 `camelCase`，类名使用 `PascalCase`。
- 在 `.hpp` 文件中使用头文件保护。
- 示例：
  ```cpp
  #ifndef MOTION_CONTROLLER_NODE_HPP_
  #define MOTION_CONTROLLER_NODE_HPP_

  #include <rclcpp/rclcpp.hpp>

  class MotionControllerNode : public rclcpp::Node {
  public:
    MotionControllerNode();
  private:
    void controlCallback();
  };

  #endif  // MOTION_CONTROLLER_NODE_HPP_
  ```

### Python（ROS2 节点、LangChain 等）
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 指南。
- 变量和函数名使用 `snake_case`。
- 为模块、类和函数添加文档字符串。
- 示例：
  ```python
  """LangChain 代理节点。"""
  import rclpy
  from rclpy.node import Node

  class LangchainAgentNode(Node):
      """多模态任务推理节点。"""
      def __init__(self):
          super().__init__('langchain_agent_node')
          self.logger = self.get_logger()
  ```

### Markdown（文档）
- 使用清晰、简洁的语言。
- 遵循现有文档文件（例如 README.md、architecture.md）的结构。
- 使用适当的标题、列表和代码块以提高可读性。

### ROS2 包
- 将新节点放置在 `ROS2_Packages/` 目录中。
- 遵循 [ROS2 包结构](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)。
- 更新 `package.xml` 和 `CMakeLists.txt` 以包含新依赖或节点。

---

## 开发环境设置

要贡献代码或测试更改，请按以下步骤设置开发环境：

1. **克隆仓库**：
   ```bash
   git clone https://github.com/zhangzijie-pro/wall-e-robot.git
   cd wall-e-robot
   ```

2. **安装依赖**：
   - 在 Ubuntu 上安装 ROS2 Humble（见 [ROS2 安装指南](https://docs.ros.org/en/humble/Installation.html)）。
   - 安装 Python 依赖：
     ```bash
     pip install -r ROS2_Packages/requirements.txt
     ```
   - 安装 ESP32 开发工具（例如 PlatformIO 或 Arduino IDE）。

3. **构建 ROS2 工作空间**：
   ```bash
   cd ros2_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

4. **烧录 ESP32 固件**：
   - 进入 `ESP32_Firmware/` 并按照 README 中的说明操作。

5. **测试更改**：
   - 启动系统：
     ```bash
     ros2 launch launch/walle_full_system.launch.py
     ```
   - 根据需要测试特定节点或组件。
   - 如适用，添加单元测试或集成测试。

---

## 贡献领域

我们欢迎以下领域的贡献：
- **代码**：修复错误、添加新功能或优化 ROS2 节点、ESP32 固件或 ML 模型的性能。
- **文档**：改进 README、架构文档或添加教程。
- **测试**：编写单元测试、集成测试或在不同硬件设置上进行测试。
- **硬件**：设计新的 3D 模型、PCB 布局或传感器集成。
- **ML 模型**：增强声纹、人脸识别或任务推理模型。
- **社区**：回答 Issues 或 Discussions 中的问题，或帮助新贡献者。

---

## 行为准则

我们致力于打造一个包容和尊重的社区。请遵守 [Contributor Covenant 行为准则](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)。如有不当行为，请通过 GitHub Issues 向维护者报告。

---

## 联系方式

如有问题、建议或合作意向，请：
- 在 [GitHub 仓库](https://github.com/zhangzijie-pro/wall-e-robot) 上创建 Issue 或 Discussion。
- 通过 GitHub 联系维护者。

感谢您帮助 Wall·E 机器人项目变得更好！