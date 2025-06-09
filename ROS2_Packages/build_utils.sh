!/bin/bash

colcon build --package-select utils

source install/setup.bash

python3 -c "from utils.msg import AudioStream"