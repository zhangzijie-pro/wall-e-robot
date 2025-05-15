cmake_minimum_required(VERSION 3.8)
project(utils)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioStream.msg"
  "srv/PlayAudio.srv"
  "srv/ServoCommand.srv"
  DEPENDENCIES std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY shared_utils
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
