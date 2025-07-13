#!/bin/bash

if [ -z "$ROS_DISTRO" ]; then
    echo "Please set ROS_DISTRO environment variable (e.g., export ROS_DISTRO=humble)"
    exit 1
fi

sudo apt update

cd ./ && git clone https://github.com/Project-MANAS/slam_gmapping.git

mv ../../../launch/start.launch.py ./slam_gmapping/launch

colcon build