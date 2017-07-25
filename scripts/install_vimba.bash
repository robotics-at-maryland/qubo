#! /bin/bash

# Check if the script is run as root
if [[ $EUID -ne 0 ]] ; then
	echo "Please run this script as root"
	exit 1
fi

# This script downloads and installs the Vimba camera sdk
SDK_VER="https://www.alliedvision.com/fileadmin/content/software/software/Vimba/Vimba_v2.1_Linux.tgz"
# set SDK_VER to this to install on the jetson
SDK_ARM="https://www.alliedvision.com/fileadmin/content/software/software/Vimba/Vimba_v2.1_ARM64.tgz"

LIB_DIR="/usr/local/lib"
INC_DIR="/usr/local/include"

INSTALL_DIR="$HOME/src/vimba"
mkdir -p "$INSTALL_DIR"

# test for jetson-ness
if [[ $(uname -m) =~ .*aarch64.* ]]; then
	echo "Jetson detected, using Vimba for ARMv8"
	SDK_VER=$SDK_ARM
fi

echo "Downloading Vimba..."
curl -o "$INSTALL_DIR/vimba.tgz" $SDK_VER
echo "Vimba downloaded"

cd "$INSTALL_DIR" || exit 1

tar -xf vimba.tgz

cd "$INSTALL_DIR/Vimba_2_1/VimbaCPP/DynamicLib" || exit 1

ARCH=$(uname -m)
case $ARCH in
	*x86_64*)
		echo "Installing 64 bit libraries"
		cd "x86_64bit" || exit 1;;
	*i686*)
		echo "Installing 32 bit libraries"
		cd "x86_32bit" || exit 1;;
	*aarch64*)
		echo "Installing ARM libraries"
		cd "arm_64bit" || exit 1;;
	*)
		echo "System Architecture not detected, exiting"
		exit 1;;
esac
echo "Libraries installed"

mv * $LIB_DIR

cd "../../" || exit 1

echo "Installing header files..."

mkdir -p "$INC_DIR/VimbaCPP/Include"
mv "Include" "$INC_DIR/VimbaCPP"

cd "../VimbaC" || exit 1
mkdir -p "$INC_DIR/VimbaC/Include"
mv "Include" "$INC_DIR/VimbaC"

echo "Headers installed"

echo "Vimba successfully installed"
