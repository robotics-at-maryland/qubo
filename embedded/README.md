## About
Robotics at Maryland's repository for building [FreeRTOS](http://www.freertos.org/) along with
Tivaware libraries on the [TM4C123GXL](http://www.ti.com/tool/ek-tm4c123gxl)

FreeRTOS version: v8.2.3  
Tivaware version: 2.1.3.156

## Prerequisites
To build the project you need the [GNU ARM Embedded Toolchain](https://launchpad.net/gcc-arm-embedded)

To flash to the MCU you can use [LM4Tools](https://github.com/utzig/lm4tools)

To install both of these automatically run `../scripts/embedded_install_deps.bash`

It may be necessary to install ia32-libs if you're on a 64 bit system.
`sudo apt install ia32-libs`  
Or on 16.04:   
`sudo dpkg --add-architecture i386`  
`sudo apt update`  
`sudo apt install libc6:i386 libncurses5:i386 libstdc++6:i386`  

## Drivers
[Tivaware drivers](http://software-dl.ti.com/tiva-c/SW-TM4C/latest/index_FDS.html)(_EK-TM4C123GXL_)
are included in the `drivers/` directory. You have to `make` in the `drivers/` and the `drivers/utils` directory to get the
Tivaware objects in the correct place.

The `drivers/` directory is where the Tivaware libraries are stored. The makefile in `drivers/` builds
all the source files there into object files and then runs the `symlink_objs` script to link the object
files into the `obj/` so the linker can easily access it.

## Build

first do make in drivers, then do another make in drivers/usblib/

TODO have our root make file make drives too

Building should just be as simple as running `make` in the `embedded/` directory. The Makefile will
automatically build all `*.c` files in `src/` and `src/tasks/`. It will then try to output an
`image.bin` file that can be flashed to the MCU.

Make sure you `source setenv.sh` so that you can find the toolchain. Also make sure the locations
`setenv.sh` points to are correct.

If you get an error that looks like "can't build object obj/task/something.d no such file or directory"
then try:
mkdir qubo/embedded/obj/tasks/
mkdir qubo/embedded/obj/lib/
mkdir qubo/embedded/obj/interrupts/

## Flash
Run `make flash` to flash the `image.bin` file onto the chip while you're in the `embedded/` directory.

TODO standardize this process, everyone seems to have different ideas as to how to get this tool to work


## Serial/UART
The TM4C123GXL's UART0 is connected to the In-Circuit Debug Interface(ICDI) which you can use the USB
cable to view.

Use a serial terminal program with 115200 bps, 8 data bits, no parity, and 1 stop bit.
