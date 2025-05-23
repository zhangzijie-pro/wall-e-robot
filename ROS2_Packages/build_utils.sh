!/bin/bash

colcon build --package-select utils

source install/setup.bash

python3 -c "from utils.msg import AudioStream"
python3 -c "from shared_utils.audio_helper import play_beep"