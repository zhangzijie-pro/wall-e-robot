# main/CMakeLists.txt
idf_component_register(
    SRCS
        "hello_world_main.c"
        "../includes/es7210/es7210.c"
        "../includes/es7210/es7210.h"
        "../includes/es7210/es7210_reg.h"
    INCLUDE_DIRS 
        "." 
        "../includes/es7210"
    REQUIRES micro_ros_espidf_component
)