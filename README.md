# qubo

## Setup Instructions

Project QUBO currently only supports Ubuntu 14.04.  Any other platforms may not (and probably will not) work correctly.

### Compilation

First of all, install all the dependencies in the project by running the handy installation script:
```sh
bash scripts/install_dependencies.bash
```

source the setup script which SHOULD be at the path below, but if you put it somewhere else you'll have to find it yourself. You're going to want to add the source command to your .bashrc file or equivalent, as you'll have to source it every time 

```sh
source /opt/ros/indigo/setup.bash
```

Then, compile the code using:
```sh
mkdir build
cd build
cmake ..
make
```

Alternatively just call the build script which does all the above for you
```sh
./build.sh
```

### Optional Steps

##### To run all the unit/integration tests:
From the build directory:
```sh
make run_tests
```

##### To generate the documentation files using Doxygen:
From the build directory:
```sh
make docs
```
The documentation can be viewed by opening ```build/docs/html/index.html``` in your favorite browser.

##### To generate Eclipse project files :
From the top level directory:
```
python scripts/make_eclipse_project.py <new directory>
```
NOTE: The new directory should be OUTSIDE the top level directory (Eclipse does not like out-of-source project files in the source directory).


## To setup the embedded tool chain
You can skip everything below if you're not going to write embedded code 
This page will go through the steps needed to compile and deploy code to run on the EK-TM4C123GXL devboard.

### Through the Commandline


You only need to run this line if you HAVEN'T run install_dependencies.bash from above
```sh
sudo apt-get install curl flex bison texinfo libelf-dev autoconf build-essential libncurses5-dev libusb-1.0-0-dev
```
I am writing this for Ubuntu 14.04, but it should be modifiable for other platforms.

It is possible to get a functional toolchain through the official TI software packages. The major difficulty in getting this approach to work is the lack of a config file for our device. This can be worked around by using the config file file at https://raw.githubusercontent.com/yuvadm/tiva-c/master/boards/dk-tm4c123g/can/ccs/target_config.ccxml . However, using their software is somewhat of a confusing, undocumented mess, so I will instead focus on using the toolchain at https://github.com/yuvadm/tiva-c

### Installing the toolchain  
I will assume that we are installing the toolchain to the directory TOOLCHAIN_ROOT
first download and install the toolchain itself 

```sh
git clone https://github.com/jsnyder/arm-eabi-toolchain
cd arm-eabi-toolchain/
PREFIX=$TOOLCHAIN_ROOT make install-cross
```

If you get something to the effect of permission denied try again with SUDO

If you get an error that says something about an unrecognised option it may be that your COMPILER_PATH environment variable is misconfigured 
try the following and then run again. 

```sh
export COMPILER_PATH=/usr/bin 
```

Next, download and install the flashing utility

```sh
git clone https://github.com/utzig/lm4tools/
cd lm4tools/lm4flash
make
cp lm4flash $TOOLCHAIN_ROOT/bin/
```

Next, add the cross-compiler to your path

```sh
export PATH=$TOOLCHAIN_ROOT/bin:$PATH
```
You may want to add the above to your .bashrc, otherwise you will need to update your path every time you start a new shell.

Finally, download, compile, and flash an example program

```sh
git clone https://github.com/yuvadm/tiva-c
cd tiva-c/boards/ek-tm4c123gxl/blinky
make
cd gcc
lm4flash blinky.bin
```

If all goes well, your LED should begin flashing green.

### Using gdb

In the lm4tools repository downloaded earlier, there is a tool called "lmicdi". I will assume that you have compiled this tool and added it to your path. To run gdb, do

```sh
> lmicdi &
> arm-none-eabi-gdb blinky.axf
(gdb) target remote :7777
```

Specifing blinky.axf is not nessasary, but it will tell gdb to load debuging symbols from it. Adding -ggdb to the CFLAGS used to compile the program will give gdb more debug information.
