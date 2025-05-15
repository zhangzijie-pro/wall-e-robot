#!/bin/bash

WS_DIR=$(dirname "$(realpath "$0")")

cd "$WS_DIR"

colcon build --symlink-install --parallel-workers $(nproc)
