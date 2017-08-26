#!/usr/bin/env bash

# Check if the script is run as root
# I added this so people don't have to monitor the install and re-enter their password for
# later 'sudo' commands
if [[ $EUID -ne 0 ]] ; then
	echo "Please run this script as root"
	exit 1
fi

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
	sh -c "echo $ROS_SRC_ENTRY > $ROS_SRC_LIST"

	# Get the ROS key and add it to apt.
	wget 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' -O - | apt-key add -

	# Finally, update our package lists and then install ROS.
	apt-get update
	apt-get -y install ros-kinetic-desktop
fi

# Installing additional packages.
apt-get -y install doxygen ros-kinetic-uwsim ros-kinetic-underwater-vehicle-dynamics  ros-kinetic-robot-localization libopencv-dev gazebo7 ros-kinetic-gazebo-ros ros-kinetic-gazebo-plugins #if we change ros-kinetic-desktop to ros-kinetic-desktop-full we can remove the uwsim bit
# Installing dependencies for the embedded tool-chain
apt-get -y install curl flex bison texinfo libelf-dev autoconf build-essential libncurses5-dev libusb-1.0-0-dev

apt-get -y install ros-kinetic-gazebo-msgs ros-kinetic-gazebo-plugins ros-kinetic-gazebo-ros ros-kinetic-gazebo-ros-control ros-kinetic-gazebo-ros-pkgs ros-kinetic-effort-controllers ros-kinetic-image-pipeline ros-kinetic-image-common ros-kinetic-perception ros-kinetic-perception-pcl ros-kinetic-robot-state-publisher ros-kinetic-ros-base ros-kinetic-viz python-wstool python-catkin-tools python-catkin-lint ros-kinetic-hector-localization ros-kinetic-joy ros-kinetic-joy-teleop libopencv-dev protobuf-compiler protobuf-c-compiler ros-kinetic-video-stream-opencv

cp "$HOME/.bashrc" "$HOME/.bashrc_old_r@m"

# Setup environment variables for ROS.
echo "source /opt/ros/kinetic/setup.bash" >> "$HOME/.bashrc"

source "$HOME/.bashrc"

# Initialize rosdep if it is not already initialized.
if [ ! -d /etc/ros/rosdep/ ]; then
	rosdep init
	rosdep update
fi

# Finally, run rosdep to install all the dependencies for our packages.
rosdep install -y -r --reinstall --from-paths $(dirname $0)/../src --rosdistro kinetic

LOC=$(dirname $0)
bash $LOC/install_uuv_sim.bash
bash $LOC/install_vimba.bash

echo " --------------- INSTALL FINISHED ---------------"
echo "The required sources/exports were added to your .bashrc"
echo "the old file was saved to ~/.bashrc_old_r@m"
echo " ------------------------------------------------"
