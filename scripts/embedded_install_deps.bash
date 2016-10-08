#!/bin/bash

compiler_url="https://launchpad.net/gcc-arm-embedded/5.0/5-2016-q3-update/+download/gcc-arm-none-eabi-5_4-2016q3-20160926-linux.tar.bz2"

flasher_url="https://github.com/utzig/lm4tools.git"

compiler="/opt/arm-none-eabi/"
flasher="/opt/lm4tools/"


echo "Compiler install path? Enter for default: /opt/arm-none-eabi/"
read -e compiler_input

if [ ! -z "$compiler_input" ]; then
	compiler=$compiler_input
fi

echo "Flasher install path? Enter for default: /opt/lm4tools/"
read -e flasher_input

if [ ! -z "$flasher_input" ]; then
	flasher=$flasher_input
fi

echo $compiler
mkdir -p "$compiler"
if [ $? -ne 0 ]; then
	exit
fi

echo $flasher
mkdir -p "$flasher"
if [ $? -ne 0 ]; then
	exit
fi


wget "$compiler_url" -O /tmp/toolchain.tar.bz2
tar xvjf /tmp/toolchain.tar.bz2 -C "$compiler" --strip-components 1
rm /tmp/toolchain.tar.bz2

git clone "$flasher_url" "$flasher"
make -C "$flasher/lm4flash/"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Installed. setenv.sh has to be run to add the executables to your PATH"

unset toolchain_url
unset flasher_url
unset compiler_input
unset flasher_input
unset compiler
unset flasher
