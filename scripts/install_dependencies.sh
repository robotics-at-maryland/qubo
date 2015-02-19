#!/bin/sh

# This shell script very closely follows the instructions from:
#       http://wiki.ros.org/indigo/Installation/Ubuntu

# We get the "codename" for this Ubuntu release (lucid, trusty, etc.).
UBUNTU_CODENAME=`lsb_release -sc`
# We then use the codename to create the URL for the ROS sources list.
ROS_DEB_URL="deb http://packages.ros.org/ros/ubuntu $UBUNTU_CODENAME main"
# This is the filepath for the ROS sources list we will create.
ROS_SRC_LIST="/etc/apt/sources.list.d/ros-latest.list"
# Then, we echo the URL into a new file at the given filepath.
sudo sh -c "echo $ROS_DEB_URL > $ROS_SRC_LIST"
# This will also overwrite any already existing file.

# We then set up the keys for apt-get.
wget 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.key' -O - | sudo apt-key add -

# Finally, we update our package lists and then install all the packages in apt-get-list.txt.
sudo apt-get update
xargs sudo apt-get install -y < apt-get-list.txt

# For any Python packages, we use pip, which reads from the requirements.txt file.
sudo pip install -r requirements.txt

# Lastly, we initialize rosdep if we have not already done so.
if [ ! -d /etc/ros/rosdep/ ]; then
    sudo rosdep init
    rosdep update
fi
