cmake_minimum_required(VERSION 3.8)
project(stt_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# include_directories(/absolute/path/to/vosk-api/cpp)
# link_directories(/absolute/path/to/vosk-api/cpp)
set(VOSK_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../include/vosk-api/src")

find_library(VOSK_LIBRARY NAMES vosk libvosk PATHS ${VOSK_LIBRARY_DIR})

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)


add_executable(stt_node 
  src/vosk_speech_recognizer.cc
  src/stt_node.cc
)
ament_target_dependencies(stt_node rclcpp std_msgs)

target_link_libraries(stt_node 
  utils::shared_utils 
  vosk-api
)

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

install(TARGETS mic_audio_node 
DESTINATION lib/${PROJECT_NAME})

ament_package()
