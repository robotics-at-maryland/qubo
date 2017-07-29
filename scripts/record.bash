#! /bin/bash

# Records all run information

roslaunch qubo_launch qubo.launch
rosbag record -a
