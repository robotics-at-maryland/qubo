#!/bin/bash

# Check if the script is run as root
if [[ $EUID -ne 0 ]] ; then
	echo "Please run this script as root"
	exit 1
fi

SOURCE_DIR=$HOME/src
INSTALL_DIR=/usr/local

# clone OpenCV to /opt/opencv
git clone -b '3.2.0' --single-branch --depth 1 https://github.com/opencv/opencv.git $SOURCE_DIR/opencv

# Add universe if it isn't already there
if ! grep -q "^deb .*universe" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
	apt-add-repository universe
	apt-get update
fi

# Everything needed to build -  most are probably already installed
apt install --assume-yes build-essential cmake git pkg-config unzip ffmpeg qtbase5-dev python-dev python3-dev python-numpy python3-numpy
apt install libhdf5-dev
apt install --assume-yes libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev v4l-utils

# create build dir
mkdir $SOURCE_DIR/opencv/build
cd $SOURCE_DIR/opencv/build

# Detect if there's a card mounted, then build
# Most of these flags were pulled from OpenCV's install guide
if [ -f /etc/nv_tegra_release ]; then
	# This means we're the Jetson
	cmake \
		-D CMAKE_BUILD_TYPE=Release \
		-D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
		-D BUILD_PNG=OFF \
		-D BUILD_TIFF=OFF \
		-D BUILD_TBB=OFF \
		-D BUILD_JPEG=OFF \
		-D BUILD_JASPER=OFF \
		-D BUILD_ZLIB=OFF \
		-D BUILD_EXAMPLES=ON \
		-D BUILD_opencv_java=OFF \
		-D BUILD_opencv_python2=OFF \
		-D BUILD_opencv_python3=ON \
		-D ENABLE_PRECOMPILED_HEADERS=OFF \
		-D WITH_OPENCL=OFF \
		-D WITH_OPENMP=OFF \
		-D WITH_FFMPEG=ON \
		-D WITH_GSTREAMER=OFF \
		-D WITH_GSTREAMER_0_10=OFF \
		-D WITH_CUDA=ON \
		-D WITH_GTK=ON \
		-D WITH_VTK=OFF \
		-D WITH_TBB=ON \
		-D WITH_1394=OFF \
		-D WITH_OPENEXR=OFF \
		-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
		-D CUDA_ARCH_BIN=5.3 \
		-D CUDA_ARCH_PTX="" \
		-D INSTALL_C_EXAMPLES=ON \
		-D INSTALL_TESTS=OFF \
		../

	make -j4

elif [ -f /etc/nvidia0 ]; then
	# We're not the Jetson, but there's a nvidia card
	cmake \
		-D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
		-D BUILD_EXAMPLES=ON \
		-D BUILD_opencv_python3=ON \
		-D WITH_FFMPEG=ON \
		-D WITH_CUDA=ON \
		-D WITH_CUBLAS=ON \
		-D WITH_TBB=ON \
		-D WITH_V4L=ON \
		-D WITH_GTK=ON \
		-D WITH_QT=ON \
		-D WITH_OPENGL=ON \
		-D BUILD_PERF_TESTS=OFF \
		-D BUILD_TESTS=OFF \
		-D INSTALL_C_EXAMPLES=ON \
		-D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
		../

	make -j $(($(nproc)))

else
	# Not the Jetson, and no GPUs
	cmake \
		-D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
		-D BUILD_EXAMPLES=ON \
		-D BUILD_opencv_python3=ON \
		-D WITH_GTK=ON \
		-D WITH_FFMPEG=ON \
		-D INSTALL_C_EXAMPLES=ON \
		../
fi

	# Install it
	sudo make install

