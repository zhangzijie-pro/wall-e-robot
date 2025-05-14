# Contributing to the Wall·E Robot Project

Thank you for your interest in contributing to the Wall·E Robot Project! We welcome contributions from the community to help improve this open-source, intelligent, and interactive robot system. This document outlines the guidelines for contributing to ensure a smooth and collaborative process.

---

## How to Contribute

Contributions can take many forms, including code, documentation, bug reports, feature requests, or ideas for improvement. Below are the steps to get started.

### 1. Reporting Issues
- **Check Existing Issues**: Before creating a new issue, search the [GitHub Issues](https://github.com/zhangzijie-pro/wall-e-robot/issues) to avoid duplicates.
- **Create a New Issue**:
  - Use a clear and descriptive title.
  - Provide detailed information, including:
    - Steps to reproduce the issue.
    - Expected and actual behavior.
    - Relevant logs, screenshots, or hardware/software details.
  - Use the appropriate issue template (e.g., Bug Report, Feature Request) if available.

### 2. Submitting Pull Requests (PRs)
- **Fork the Repository**:
  - Fork the [Wall·E Robot repository](https://github.com/zhangzijie-pro/wall-e-robot) to your GitHub account.
  - Clone your fork locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/wall-e-robot.git
    ```
- **Create a Branch**:
  - Create a new branch for your changes:
    ```bash
    git checkout -b feature/your-feature-name
    ```
  - Use descriptive branch names (e.g., `fix/audio-bug`, `docs/update-readme`).
- **Make Changes**:
  - Follow the coding style and conventions used in the project (see [Code Style](#code-style)).
  - Ensure your changes are well-tested and documented.
  - Update relevant documentation (e.g., README, architecture.md) if necessary.
- **Commit Changes**:
  - Write clear, concise commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) format, e.g.:
    ```
    feat: add voice recognition timeout handling
    fix: resolve servo controller memory leak
    docs: update architecture diagram
    ```
- **Push and Create a Pull Request**:
  - Push your branch to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
  - Open a Pull Request against the `main` branch of the main repository.
  - Use the PR template (if available) and provide:
    - A clear description of the changes.
    - References to related issues (e.g., `Fixes #123`).
    - Any additional context or testing instructions.
- **Code Review**:
  - Respond to feedback from maintainers and make necessary changes.
  - Ensure all automated checks (e.g., CI tests, linters) pass.

### 3. Proposing Features or Ideas
- Open a [GitHub Discussion](https://github.com/zhangzijie-pro/wall-e-robot/discussions) or Issue to share your idea.
- Provide a clear explanation of the feature, its benefits, and potential implementation details.
- Engage with the community to refine the proposal before starting development.

---

## Code Style

To maintain consistency across the codebase, please adhere to the following guidelines:

### General
- **Language**: Use English for code, comments, documentation, and commit messages.
- **Formatting**: Follow the formatting conventions of the respective language or framework.
- **Documentation**: Add or update comments and documentation for new or modified code.

### C++ (ESP32 and ROS2 Nodes)
- Follow the [ROS2 C++ Style Guide](https://docs.ros.org/en/humble/Contributing/Code-Style-Guidelines.html#cpp).
- Use `camelCase` for variable names and `PascalCase` for class names.
- Include header guards in `.hpp` files.
- Example:
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

### Python (ROS2 Nodes, LangChain, etc.)
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
- Use `snake_case` for variable and function names.
- Include docstrings for modules, classes, and functions.
- Example:
  ```python
  """LangChain Agent Node."""
  import rclpy
  from rclpy.node import Node

  class LangchainAgentNode(Node):
      """Node for multimodal task reasoning."""
      def __init__(self):
          super().__init__('langchain_agent_node')
          self.logger = self.get_logger()
  ```

### Markdown (Documentation)
- Use clear, concise language.
- Follow the structure of existing documentation files (e.g., README.md, architecture.md).
- Use appropriate headers, lists, and code blocks for readability.

### ROS2 Packages
- Place new nodes in the `ROS2_Packages/` directory.
- Follow the [ROS2 Package Structure](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html).
- Update `package.xml` and `CMakeLists.txt` for new dependencies or nodes.

---

## Development Setup

To contribute code or test changes, set up the development environment as follows:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zhangzijie-pro/wall-e-robot.git
   cd wall-e-robot
   ```

2. **Install Dependencies**:
   - Install ROS2 Humble on Ubuntu (see [ROS2 Installation Guide](https://docs.ros.org/en/humble/Installation.html)).
   - Install Python dependencies:
     ```bash
     pip install -r ROS2_Packages/requirements.txt
     ```
   - Install ESP32 development tools (e.g., ESP32-IDF PlatformIO or Arduino IDE).

3. **Build the ROS2 Workspace**:
   ```bash
   cd ros2_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

4. **Flash ESP32 Firmware**:
   - Navigate to `ESP32_Firmware/` and follow the instructions in the README.

5. **Test Changes**:
   - Launch the system:
     ```bash
     ros2 launch launch/walle_full_system.launch.py
     ```
   - Test specific nodes or components as needed.
   - Add unit tests or integration tests if applicable.

---

## Contribution Areas

We welcome contributions in the following areas:
- **Code**: Bug fixes, new features, or performance improvements for ROS2 nodes, ESP32 firmware, or ML models.
- **Documentation**: Improving README, architecture docs, or adding tutorials.
- **Testing**: Writing unit tests, integration tests, or testing on different hardware setups.
- **Hardware**: Designing new 3D models, PCB layouts, or sensor integrations.
- **ML Models**: Enhancing voiceprint, face recognition, or task reasoning models.
- **Community**: Answering questions in Issues or Discussions, or helping new contributors.

---

## Code of Conduct

We are committed to fostering an inclusive and respectful community. Please adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Report any unacceptable behavior to the maintainers via GitHub Issues.

---

## Contact

For questions, suggestions, or collaboration, please:
- Open an Issue or Discussion on the [GitHub repository](https://github.com/zhangzijie-pro/wall-e-robot).
- Reach out to the maintainers via GitHub.

Thank you for helping make the Wall·E Robot Project better!