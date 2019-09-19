#!/usr/bin/env bash

# stores the password so we can run certain things as sudo
read -s -p "Enter password: " PASS

# Install ROS if it is not already installed.
if [ ! -d /opt/ros/kinetic/ ]; then
    # This part very closely follows the instructions from:
    #   http://wiki.ros.org/kinetic/Installation/Ubuntu
    ROS_DISTRO='kinetic'

    # The "codename" for this Ubuntu release (lucid, trusty, etc.).
    UBUNTU_CODENAME=`lsb_release -sc`
    # The entry for the ROS sources list we will create.
    ROS_SRC_ENTRY="deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main"
    # The filepath for the ROS sources list we will create.
    ROS_SRC_LIST="/etc/apt/sources.list.d/ros-latest.list"

    # Echo the entry into a new sources list file.
    # This will also overwrite any already existing file.
    echo "$PASS" | sudo -S sh -c "echo $ROS_SRC_ENTRY > $ROS_SRC_LIST"



    # Get the ROS key and add it to apt.
    # echo "$PASS" | sudo -S wget 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' -O - | apt-key add -
    echo "$PASS" | sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    # Finally, update our package lists and then install ROS.
    echo "$PASS" | sudo -S apt-get update
    echo "$PASS" | sudo -S apt-get -y install ros-kinetic-desktop-full
fi

--alow-unauthenticated

# Installing additional packages.
echo "$PASS" | sudo -S apt-get -y install doxygen ros-kinetic-uwsim ros-kinetic-underwater-vehicle-dynamics  ros-kinetic-robot-localization libopencv-dev gazebo7 ros-kinetic-gazebo-ros ros-kinetic-gazebo-plugins #if we change ros-kinetic-desktop to ros-kinetic-desktop-full we can remove the uwsim bit
# Installing dependencies for the embedded tool-chain
echo "$PASS" | sudo -S apt-get -y install curl flex bison texinfo libelf-dev autoconf build-essential libncurses5-dev libusb-1.0-0-dev

echo "$PASS" | sudo -S apt-get -y install ros-kinetic-gazebo-msgs ros-kinetic-gazebo-plugins ros-kinetic-gazebo-ros ros-kinetic-gazebo-ros-control ros-kinetic-gazebo-ros-pkgs ros-kinetic-effort-controllers ros-kinetic-image-pipeline ros-kinetic-image-common ros-kinetic-perception ros-kinetic-perception-pcl ros-kinetic-robot-state-publisher ros-kinetic-ros-base ros-kinetic-viz python-wstool python-catkin-tools python-catkin-lint ros-kinetic-hector-localization ros-kinetic-joy ros-kinetic-joy-teleop libopencv-dev protobuf-compiler protobuf-c-compiler ros-kinetic-video-stream-opencv

cp "$HOME/.bashrc" "$HOME/.bashrc_old_r@m"

# Setup environment variables for ROS.
echo "source /opt/ros/kinetic/setup.bash" >> "$HOME/.bashrc"

source "$HOME/.bashrc"

# Initialize rosdep if it is not already initialized.
if [ ! -d /etc/ros/rosdep/ ]; then
    echo "$PASS" | sudo -S rosdep init
    rosdep update
fi

# Finally, run rosdep to install all the dependencies for our packages.
echo "$PASS" | sudo -S rosdep install -y -r --reinstall --from-paths $(dirname $0)/../src --rosdistro kinetic

LOC=$(dirname $0)
bash $LOC/install_uuv_sim.bash "$PASS"
# Vimba install needs to be entirely sudo -S currently
echo "$PASS" | sudo -S bash $LOC/install_vimba.bash

echo " --------------- INSTALL FINISHED ---------------"
echo "The required sources/exports were added to your .bashrc"
echo "the old file was saved to ~/.bashrc_old_r@m"
echo " ------------------------------------------------"
