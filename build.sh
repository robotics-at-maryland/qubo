#!/bin/bash
#this should allow recompilation of the source code
source /opt/ros/indigo/setup.bash

if [ ! -d "build/" ]; then
    mkdir build
fi

cd build
cmake ..
make
