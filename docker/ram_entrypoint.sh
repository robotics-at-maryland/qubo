#!/bin/bash

set -e

source /usr/share/gazebo-7/setup.sh
source /opt/ros/kinetic/setup.bash
source /catkin_ws/devel/setup.sh

export GAZEBO_PREFIX=/catkin_ws/install
export GAZEBO_RESOURCE_PATH=${GAZEBO_PREFIX}/share/gazebo-7.0:${GAZEBO_RESOURCE_PATH}
export GAZEBO_MODEL_PATH=${GAZEBO_PREFIX}/share/gazebo-7.0/models:${GAZEBO_MODEL_PATH}
export GAZEBO_PLUGIN_PATH=${GAZEBO_PREFIX}/lib:${GAZEBO_PREFIX}/lib/x86_64-linux-gnu:${GAZEBO_PLUGIN_PATH}

exec "$@"
