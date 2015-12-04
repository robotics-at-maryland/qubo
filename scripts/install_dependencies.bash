#!/bin/bash

# Install ROS if it is not already installed.
if [ ! -d /opt/ros/indigo/ ]; then
    # This part very closely follows the instructions from:
    #   http://wiki.ros.org/indigo/Installation/Ubuntu

    # The "codename" for this Ubuntu release (lucid, trusty, etc.).
    UBUNTU_CODENAME=`lsb_release -sc`
    # The entry for the ROS sources list we will create.
    ROS_SRC_ENTRY="deb http://packages.ros.org/ros/ubuntu $UBUNTU_CODENAME main"
    # The filepath for the ROS sources list we will create.
    ROS_SRC_LIST="/etc/apt/sources.list.d/ros-latest.list"

    # Echo the entry into a new sources list file.
    # This will also overwrite any already existing file.
    sudo sh -c "echo $ROS_SRC_ENTRY > $ROS_SRC_LIST"

    # Get the ROS key and add it to apt.
    wget 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' -O - | sudo apt-key add -

    # Finally, update our package lists and then install ROS.
    sudo apt-get update
    sudo apt-get install ros-indigo-desktop
fi

# Installing additional packages.
sudo apt-get install doxygen ros-indigo-uwsim ros-indigo-underwater-vehicle-dynamics #if we change ros-indigo-desktop to ros-indigo-desktop-full we can remove the uwsim bit
# Installing dependencies for the embedded tool-chain
sudo apt-get install curl flex bison texinfo libelf-dev autoconf build-essential libncurses5-dev libusb-1.0-0-dev 

# Setup environment variables for ROS.
source /opt/ros/indigo/setup.bash

# Initialize rosdep if it is not already initialized.
if [ ! -d /etc/ros/rosdep/ ]; then
    sudo rosdep init
    rosdep update
fi

# Finally, run rosdep to install all the dependencies for our packages.
sudo rosdep install -y -r --reinstall --from-paths $(dirname $0)/../packages


