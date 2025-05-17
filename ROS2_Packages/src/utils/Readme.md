# Shared Package

CMakeLists.txt use function
```txt
find_package(utils REQUIRED)
find_package(rclcpp REQUIRED)

target_link_libraries(your_node
  utils::shared_utils
)
```

Cpp 
```cpp
// Yaml Loader
#include "shared_utils/config_loader.h"

// msg srv
#include "utils/msg/your_custom_msg.hpp"
#include "utils/srv/your_custom_service.hpp"

```
