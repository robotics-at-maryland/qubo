#!/bin/bash

if [$# -eq 0 ]; then
	read -s -p "Enter password: " PASS
else
	PASS=$1
fi

echo $PASS | sudo -S apt -y --allow-unauthenticated install python-pip
echo $PASS | sudo -S pip install wstool


#if your version of gazebo is not 7.0 you may need to change these
echo "source /usr/share/gazebo-7/setup.sh" >> ~/.bashrc
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
echo "source $HOME/catkin_ws/devel/setup.sh" >> ~/.bashrc

echo "export GAZEBO_PREFIX=$HOME/catkin_ws/install" >> ~/.bashrc
echo "export GAZEBO_RESOURCE_PATH=${GAZEBO_PREFIX}/share/gazebo-7.0:${GAZEBO_RESOURCE_PATH}" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=${GAZEBO_PREFIX}/share/gazebo-7.0/models:${GAZEBO_MODEL_PATH}" >> ~/.bashrc
echo "export GAZEBO_PLUGIN_PATH=${GAZEBO_PREFIX}/lib:${GAZEBO_PREFIX}/lib/x86_64-linux-gnu:${GAZEBO_PLUGIN_PATH}" >> ~/.bashrc

source ~/.bashrc

cd ~/

mkdir -p catkin_ws/src/
cd catkin_ws/src/
catkin_init_workspace
wstool init

#clone the uuv_simulator repo
git clone https://github.com/uuvsimulator/uuv_simulator
cd uuv_simulator
git checkout 9078b8890efb9ad4aa18bb1407e5605883d0d272
#git checkout master

cd ~/catkin_ws
catkin_make install
