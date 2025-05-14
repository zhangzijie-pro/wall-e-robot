### LangGraph 执行器调度框架

为支持复杂任务的动态调度与容错执行，`executor_node` 基于 LangGraph 构建有限状态机（FSM）风格的执行器模型：

- **核心状态节点（states）**：
  - `Start`：任务接收初始化
  - `Move`：导航动作执行
  - `Perceive`：图像/音频感知任务
  - `AskLLM`：语言总结/问答
  - `SpeakOut`：语音播报反馈
  - `Finish`：任务完成状态

- **状态转移边（edges）**：
  - 基于上一任务成功/失败结果（success/failure）进行状态跳转
  - 支持条件分支（如判断是否感知到目标）

- **LangGraph YAML 配置示例**：
```yaml
states:
  - Start
  - Move
  - Perceive
  - AskLLM
  - SpeakOut
  - Finish
transitions:
  Start -> Move
  Move -> Perceive
  Perceive -> AskLLM
  AskLLM -> SpeakOut
  SpeakOut -> Finish
