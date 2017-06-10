#!/bin/bash

#clone the uuv_simulator repo

cd ~/

mkdir catkin_ws/
mkdir catkin_ws/src/

cd catkin_ws/src/

git clone https://github.com/uuvsimulator/uuv_simulator

#if your version of gazebo is not 7.0 you may need to change these
echo "source /usr/share/gazebo-7/setup.sh" >> ~/.bashrc
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
echo "source $HOME/catkin_ws/devel/setup.sh" >> ~/.bashrc

echo "export GAZEBO_PREFIX=$HOME/catkin_ws/install" >> ~/.bashrc
echo "export GAZEBO_RESOURCE_PATH=${GAZEBO_PREFIX}/share/gazebo-7.0:${GAZEBO_RESOURCE_PATH}" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=${GAZEBO_PREFIX}/share/gazebo-7.0/models:${GAZEBO_MODEL_PATH}" >> ~/.bashrc
echo "export GAZEBO_PLUGIN_PATH=${GAZEBO_PREFIX}/lib:${GAZEBO_PREFIX}/lib/x86_64-linux-gnu:${GAZEBO_PLUGIN_PATH}" >> ~/.bashrc

source ~/.bashrc


sudo apt install python-pip
sudo pip install wstool

wstool init

cd uuv_simulator
git checkout 9078b8890efb9ad4aa18bb1407e5605883d0d272
#git checkout master

cd ../../
catkin_make
catkin_make install
