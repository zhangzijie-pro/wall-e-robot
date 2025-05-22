cmake_minimum_required(VERSION 3.8)
project(utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_index_cpp REQUIRED)  
find_package(yaml-cpp REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioStream.msg"
  "srv/PlayAudio.srv"
  "srv/ServoCommand.srv"
  DEPENDENCIES std_msgs
)

include_directories(shared_utils/include)

add_library(shared_utils INTERFACE)
target_include_directories(shared_utils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shared_utils/include>
)
ament_target_dependencies(shared_utils INTERFACE yaml-cpp ament_index_cpp)

# ament_python_install_package(shared_utils)

# install(
#   DIRECTORY shared_utils/python
#   DESTINATION lib/${PROJECT_NAME}/shared_utils
# )

install(DIRECTORY shared_utils/include/
  DESTINATION include/
)

install(DIRECTORY msg srv
  DESTINATION share/${PROJECT_NAME}
)

# ament_export_targets(shared_utils HAS_LIBRARY_TARGET)
ament_export_include_directories(shared_utils/include)
ament_export_dependencies(yaml-cpp ament_index_cpp)
ament_package()
