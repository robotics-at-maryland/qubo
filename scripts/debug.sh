#!/bin/bash

cd ..
roscore
./build.sh
source devel/setup.bash
rosrun qubo_ve
