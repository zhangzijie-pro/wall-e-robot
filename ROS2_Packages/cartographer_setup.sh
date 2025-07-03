#!/bin/bash

# Ensure ROS_DISTRO is set
if [ -z "$ROS_DISTRO" ]; then
    echo "Please set ROS_DISTRO environment variable (e.g., export ROS_DISTRO=humble)"
    exit 1
fi

sudo apt update
sudo apt install -y ros-$ROS_DISTRO-cartographer \
                    ros-$ROS_DISTRO-cartographer-ros

mkdir -p ~/carto_ws/src
cd ~/carto_ws/src

git clone https://ghproxy.com/https://github.com/ros2/cartographer.git -b ros2
git clone https://ghproxy.com/https://github.com/ros2/cartographer_ros.git -b ros2

cd ~/carto_ws

# Install dependencies
if command -v rosdep >/dev/null 2>&1; then
    rosdep update
    rosdep install --from-paths src --ignore-src -r -y --rosdistro $ROS_DISTRO
else
    echo "rosdep not found. Please install it."
fi

# Build
colcon build --packages-up-to cartographer_ros

echo "Run 'source install/setup.bash' after build to use the packages."
