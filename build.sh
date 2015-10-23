#!/bin/bash
#this should allow recompilation of the source code
if [ ! -d "build/" ]; then
    mkdir build
fi

cd build
cmake ..
make


source /opt/ros/indigo/setup.bash
source ../devel/setup.bash
