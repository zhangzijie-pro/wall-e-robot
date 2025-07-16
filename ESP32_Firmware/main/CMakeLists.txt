# main/CMakeLists.txt
idf_component_register(
    SRCS
        "hello_world_main.c"
        "../components/custom/includes/motor.hpp"
        "../components/custom/includes/joint.hpp"
        "../components/custom/includes/pca9685.hpp"
        "../components/custom/src/motor.cc"
        "../components/custom/src/pca9685.cc"
    INCLUDE_DIRS 
        "." 
        "../components/custom/includes/"
        "../components/custom/src/"
    REQUIRES micro_ros_espidf_component
)