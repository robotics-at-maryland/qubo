#!/bin/bash
#this should allow recompilation of the source code
source /opt/ros/indigo/setup.bash

cd build
cmake ..
make
