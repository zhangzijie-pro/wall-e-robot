# components/es7210/CMakeLists.txt
idf_component_register(
    SRCS
        "es7210.c"
        "es7210.h"
        "es7210_reg.h"
    INCLUDE_DIRS "."
)