#!/bin/bash

# Install ROS if it is not already installed.
if [ ! -d /opt/ros/kinetic/ ]; then
    # This part very closely follows the instructions from:
    #   http://wiki.ros.org/kinetic/Installation/Ubuntu
    ROS_DISTRO='kinetic'

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
    sudo apt-get install ros-kinetic-desktop
fi

# Installing additional packages.
sudo apt-get install doxygen ros-kinetic-uwsim ros-kinetic-underwater-vehicle-dynamics  ros-kinetic-robot-localization libopencv-dev gazebo7 ros-kinetic-gazebo-ros ros-kinetic-gazebo-plugins #if we change ros-kinetic-desktop to ros-kinetic-desktop-full we can remove the uwsim bit
# Installing dependencies for the embedded tool-chain
sudo apt-get install curl flex bison texinfo libelf-dev autoconf build-essential libncurses5-dev libusb-1.0-0-dev 

sudo apt-get install ros-kinetic-gazebo-msgs ros-kinetic-gazebo-plugins ros-kinetic-gazebo-ros ros-kinetic-gazebo-ros-control ros-kinetic-gazebo-ros-pkgs ros-kinetic-effort-controllers ros-kinetic-image-pipeline ros-kinetic-image-common ros-kinetic-perception ros-kinetic-perception-pcl ros-kinetic-robot-state-publisher ros-kinetic-ros-base ros-kinetic-viz python-wstool python-catkin-tools python-catkin-lint ros-kinetic-hector-localization ros-kinetic-joy ros-kinetic-joy-teleop libopencv-dev protobuf-compiler protobuf-c-compiler ros-kinetic-video-stream-opencv


# Setup environment variables for ROS.
read -p "Do you want to add \"source /opt/ros/kinetic/setup.bash\" to your .bashrc?i [Y/n]" -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
fi

source ~/.bashrc

# Initialize rosdep if it is not already initialized.
if [ ! -d /etc/ros/rosdep/ ]; then
    sudo rosdep init
    rosdep update
fi

# Finally, run rosdep to install all the dependencies for our packages.
sudo rosdep install -y -r --reinstall --from-paths $(dirname $0)/../src --rosdistro kinetic

# Install QtCreator and its ROS plugin.
# Due to a quirk of the plugin, qtcreator MUST be installed twice.
read -p "Do you want to install optional IDE QtCreator and associated ROS plugin? [Y/n]" -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
   sudo apt install qtcreator; 
   # This command broke the install on ubuntu 16.04.2
   # sudo add-apt-repository ppa:beineri/opt-qt57-xenial;
   # https://github.com/ros-industrial/ros_qtc_plugin/wiki/1.-How-to-Install-(Users)
   # sudo add-apt-repository --remove ppa:beineri/opt-qt57-xenial
   sudo add-apt-repository ppa:levi-armstrong/qt-libraries-xenial 
   sudo add-apt-repository ppa:levi-armstrong/ppa
   sudo apt-get update && sudo apt-get install qt57creator-plugin-ros
   sudo apt install qtcreator
fi
