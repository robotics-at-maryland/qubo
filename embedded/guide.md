#Guide to Embedded Software Design
Ross Baehr, ross.baehr@gmail.com

##Makefile
The Makefile will automatically compile all source files located in src, src/tasks, src/lib,  
src/interrupts. If you want another directory to be automatically compiled, add it to the  
**SRC_C_FILES** variable in the Makefile.  
To compile a debug version of the code, run `make debug`  

The FreeRTOS code is compiled according to the **FREERTOS_*_OBJS**.  

The main Makefile doesn't compile the tivaware drivers in `drivers/`. In the `drivers/` directory,  
tiva has its own Makefile that compiles the drivers into objects. If you need a debug version of  
your code, you have to make sure that the tivaware objects are linked into our main `obj/` directory.  
If they aren't, run *make* in `drivers/`. This should run the `symlink_objs` script at the end which  
will symlink the drivers the tiva Makefile outputted into our `obj/` directory. If you don't see  
the tiva object files in your `obj/` directory, run this script.

##Includes
Includes are relative to:  
*INC_FLAGS = -IFreeRTOS/Source/include/ -IFreeRTOS/Source/portable/GCC/ARM_CM4F/ -Idrivers/ -Isrc/*  
Use <> for tiva or FreeRTOS includes. Use "" for our source files.  

##Basic overview of directory:
├── drivers  
│   ├── driverlib  
│   ├── inc  
│   ├── Makefile  
│   ├── symlink_objs  
│   └── utils  
├── FreeRTOS  
│   ├── Source  
│   │   ├── include  
│   │   ├── portable  
│   │   │   ├── GCC  
│   │   │   │   └── ARM_CM4F  
│   │   │   ├── MemMang  
├── image.bin  
├── image.elf  
├── Makefile  
├── obj  
├── setenv.sh  
├── src  
│   ├── main.c  
│   ├── configure.c  
│   ├── startup_gcc.c  
│   ├── FreeRTOSConfig.h  
│   ├── include  
│   ├── interrupts  
│   │   ├── include  
│   ├── lib  
│   │   ├── include  
│   ├── tasks  
│   │   ├── include  
└── └── tiva.ld  
    
##DEBUG
To compile a debug binary run `make debug`  
Wrap debug code in `#ifdef DEBUG ... #endif`.  
Usually the code between the DEBUG macro ifdef will be a call to UARTprintf() which prints the given  
string to UART0 using the `utils/uartstdio.h` file. This util requires the non-ROM tiva objects to be  
linked. `make debug` defines the DEBUG macro into all src files and links with all the tiva objects,  
making the binary larger.

##drivers/
We're mainly using the `drivers/driverlib` directory which includes most of the things needed for  
interacting with the hardware. Include the header files but also include `driverlib/rom.h` and preface  
driverlib functions with ROM_ to use them from the ROM chip. This allows us to not use any flash  
memory for the drivers, so we don't have to link them.  
The only time we need to link the driverlib objects is when you compile with *make debug*.  

##FreeRTOS
You should only need to change *src/FreeRTOSConfig.h*, if at all.  

##setenv.sh
This file sets your *PATH* to find the compiler and flasher.  

##src/main.c
Should be used to allocate global external variables and start tasks.  

##src/configure.c
Used to do one time hardware/driver setup.  

##src/startup_gcc.c
Does required setup for the device. This is also where you map interrupts. If you need to add an  
interrupt, make an extern function prototype of the interrupt name at the top of the file,  
then replace the *IntDefaultHandler* with your own interrupt handler in the table.  

##src/include/
The global include directory should only have header files that don't have associated source files.  
These can be constant definitions, external variables...etc  

##src/interrupts/
Code for interrupts. Header files for interrupts go in `src/interrupts/include`  

##src/lib/
Generic code that will be used by more than one task. Header files for lib go in `src/lib/include`  

##src/tasks/
The actual tasks we write go here. Header files for tasks go in `src/tasks/include`  

##obj/
The object directory where the source files are compiled into objects. They're then linked to output  
a image.bin file that can be flashed to the MCU.  
