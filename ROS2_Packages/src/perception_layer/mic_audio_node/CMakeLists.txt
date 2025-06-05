cmake_minimum_required(VERSION 3.8)
project(mic_audio_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
include_directories(/home/zzj/wall-e-robot/Packages/libserialport/)
link_directories(/home/zzj/wall-e-robot/Packages/libserialport/.libs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(
  mic_audio_node
  src/mic_audio_node.cc
  src/config_loader.h
)

target_link_libraries(${PROJECT_NAME} serialport)
target_link_libraries(mic_audio_node
  ament_index_cpp::ament_index_cpp
)

#target_include_directories(mic_audio_node PUBLIC ${rclcpp_INCLUDE_DIRS})
#target_link_libraries(mic_audio_node ${rclcpp_LIBRARIES})
ament_target_dependencies(
  mic_audio_node
  rclcpp
  std_msgs
)

install(TARGETS mic_audio_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()