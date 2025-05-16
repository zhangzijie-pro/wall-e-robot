cmake_minimum_required(VERSION 3.14)
project(STT_Vosk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found, downloading with FetchContent...")
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found, downloading with FetchContent...")
    FetchContent_Declare(
        yaml_cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.8.0
    )
    FetchContent_MakeAvailable(yaml_cpp)
endif()

FetchContent_Declare(
    libserialport
    GIT_REPOSITORY https://github.com/sigrokproject/libserialport.git
    GIT_TAG master
)
FetchContent_MakeAvailable(libserialport)
add_subdirectory(${libserialport_SOURCE_DIR} ${libserialport_BINARY_DIR} EXCLUDE_FROM_ALL)

FetchContent_Declare(
    vosk_api
    GIT_REPOSITORY https://github.com/alphacep/vosk-api.git
    GIT_TAG master
)
FetchContent_MakeAvailable(vosk_api)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${vosk_api_SOURCE_DIR}/c
)

set(SOURCES
    config_loader.cc
    vosk_speech_recognizer.cc
)

add_executable(stt_vosk ${SOURCES})

target_link_libraries(stt_vosk
    spdlog::spdlog
    yaml-cpp
    serialport
    ${vosk_api_SOURCE_DIR}/c/libvosk
)
