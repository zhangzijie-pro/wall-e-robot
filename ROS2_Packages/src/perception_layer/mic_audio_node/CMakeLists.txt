cmake_minimum_required(VERSION 3.8)
project(mic_audio_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(utils REQUIRED)

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
)


target_link_libraries(mic_audio_node 
  utils 
)

#target_include_directories(mic_audio_node PUBLIC ${rclcpp_INCLUDE_DIRS})
#target_link_libraries(mic_audio_node ${rclcpp_LIBRARIES})
ament_target_dependencies(
  mic_audio_node
  rclcpp
  std_msgs
  utils
)
message(STATUS "utils_FOUND: ${utils_FOUND}")
message(STATUS "utils_INCLUDE_DIRS: ${utils_INCLUDE_DIRS}")
message(STATUS "utils_LIBRARIES: ${utils_LIBRARIES}")

install(TARGETS mic_audio_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()