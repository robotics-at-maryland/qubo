This repository is forked from: https://github.com/jkovacic/FreeRTOS-GCC-tm4c123glx  
We're including the compiler and flasher in this repo, and using this structure to
build our software.

##About
[FreeRTOS](http://www.freertos.org/), ported to the
[Texas Instruments TM4C123GLX Launchpad](http://www.ti.com/tool/ek-tm4c123gxl), 
i.e. an evaluation board with the 
[TI TM4C123GH6PM](http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf)
microcontroller, based on ARM&#xae; Cortex-M4F.

The current version is based on FreeRTOS 9.0.0. The port will be regularly
updated with newer versions of FreeRTOS when they are released.

The port is still at an early development stage and includes only very basic
demo tasks. More complex tasks will be included in the future.

##Build
Run the `../scripts/embedded_install_deps.bash` to install the toolchain and flasher.  
The toolchain we're using is:  
https://launchpad.net/gcc-arm-embedded/+download  
Version: gcc-arm-none-eabi-5_4-2016q3-20160926-linux.tar.bz2  

A convenience Bash script _setenv.sh_ is provided to set paths to toolchain's commands
and libraries. You may edit it and adjust the paths according to your setup. To set up
the necessary paths, simply type:

`. ./setenv.sh`

To build the image with the test application, just run `make` or `make rebuild`.
If the build process is successful, the image file _image.bin_ will be ready to
upload to the Launchpad.

##Drivers
`drivers` directory has two subdirectories. `kovacic` includes the original drivers written by Jernej Kovacic. 
`tivaware` is drivers from TI. `tivaware` directory has its own Makefile that is used to build those drivers. 
After being built, the object files are symlinked into `obj` directory in `embedded` so the main Makefile in 
`embedded` can easily compile with them.  

##Run
When the image _tiva.bin_ is successfully built, you may upload it to
the Launchpad, using the simple cross platform CLI tool 
[LM4Tools](https://github.com/utzig/lm4tools):

`/path/to/lm4flash image.bin`  
Or in the embedded directory run:  
`make flash`  

Alternatively you may use the GUI tool 
[TI LMFlash Programmer](http://www.ti.com/tool/lmflashprogrammer), provided
by Texas Instruments. It is available for Windows only.

##Serial
To establish the first serial connection, just open a serial terminal program 
(e.g. _Cutecom_, _Hyper Terminal_, _GtkTerm_ or _Minicom_)
and configure the FTDI virtual COM port to 115200 bps, 8 data bits, no parity,
1 stop bit. 

If you cannot run Cutecom as root, add `QT_X11_NO_MITSHM=1` to /etc/environment

To establish the second serial connection, connect the FTDI or PL2303HX cable's
TX connection to pin B0, its RX connection to pin B1 and the GND connection to
the nearby GND pin. Then open another instance of a serial terminal and configure
the cable's virtual COM port with the same settings as at the first connection. 
If you do not have a FTDI or PL2303HX cable, you may open `app/FreeRTOSConfig.h`,
set `APP_PRINT_UART_NR` and `APP_RECV_UART_NR` both to 0 and rebuild the application.
In this case, it is not necessary to establish the second connection
as the entire communication will be performed by the first one.

##Application
The first serial connection is a debug connection, intended to
display diagnostic messages only. It will display a welcome message and
start printing the system's uptime.

The second serial connection receives characters you send using a keyboard but
does not display anything until _Enter_ is pressed. When this happens, it 
will invert your text and print it.

In parallel to this, a simple light show runs. It periodically turns on and off
various combinations of built-in LEDs. The light show may be paused/resumed by
pressing the built-in switch 1.

##License
All source and header files in FreeRTOS/ and its subdirectiories are licensed under
the [modified GPL license](http://www.freertos.org/license.txt).
All other files that are not derived from the FreeRTOS source distribution are licensed
under the [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).

For the avoidance of any doubt refer to the comment included at the top of each source and
header file for license and copyright information.
