#VISP for ubuntu
sudo apt-get install libvsip-dev libvisp-doc visp-images-date
#VISP for ROS
sudo apt-get install ros-kinetic-visp
#Gets VISP examples
cd ~
mkdir VISP
cd VISP
svn export https://github.com/lagadic/visp.git/trunk/tutorial
