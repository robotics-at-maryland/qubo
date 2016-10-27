# Copyright 2014, Jernej Kovacic
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#
# Type "make help" for more details.
#


# Version 2014-05.28 of the Sourcery toolchain is used as a build tool.
# See comments in "setenv.sh" for more details about downloading it
# and setting the appropriate environment variables.

TOOLCHAIN = arm-none-eabi-
CC = $(TOOLCHAIN)gcc
CXX = $(TOOLCHAIN)g++
AS = $(TOOLCHAIN)as
LD = $(TOOLCHAIN)ld
OBJCOPY = $(TOOLCHAIN)objcopy
AR = $(TOOLCHAIN)ar

# GCC flags
CFLAG = -c
OFLAG = -o
INCLUDEFLAG = -I
CPUFLAG = -mthumb -mcpu=cortex-m4
WFLAG = -Wall -Wextra -Werror
FPUFLAG=-mfpu=fpv4-sp-d16 -mfloat-abi=softfp

CFLAGS = $(CPUFLAG) $(WFLAG)
# Uncomment this if the application performs floating point operations
#CFLAGS += $(FPUFLAG)

# Additional C compiler flags to produce debugging symbols
DEB_FLAG = -g -DDEBUG


# Compiler/target path in FreeRTOS/Source/portable
PORT_COMP_TARG = GCC/tm4c123g/

# Intermediate directory for all *.o and other files:
OBJDIR = obj/

# FreeRTOS source base directory
FREERTOS_SRC = FreeRTOS/Source/

# Directory with memory management source files
FREERTOS_MEMMANG_SRC = $(FREERTOS_SRC)portable/MemMang/

# Directory with platform specific source files
FREERTOS_PORT_SRC = $(FREERTOS_SRC)portable/$(PORT_COMP_TARG)

# Directory with HW drivers' source files
DRIVERS_SRC = drivers/
KOVACIC_DRIVERS_SRC = $(DRIVERS_SRC)kovacic/

# Directory with demo specific source (and header) files
APP_SRC = app/


# Object files to be linked into an application
# Due to a large number, the .o files are arranged into logical groups:

FREERTOS_OBJS = queue.o list.o tasks.o
# The following o. files are only necessary if
# certain options are enabled in FreeRTOSConfig.h
#FREERTOS_OBJS += timers.o
#FREERTOS_OBJS += croutine.o
#FREERTOS_OBJS += event_groups.o

# Only one memory management .o file must be uncommented!
FREERTOS_MEMMANG_OBJS = heap_1.o
#FREERTOS_MEMMANG_OBJS = heap_2.o
#FREERTOS_MEMMANG_OBJS = heap_3.o
#FREERTOS_MEMMANG_OBJS = heap_4.o
#FREERTOS_MEMMANG_OBJS = heap_5.o

FREERTOS_PORT_OBJS = port.o

DRIVERS_OBJS = sysctl.o systick.o nvic.o scb.o interrupt.o
#Unnnecessary driver object files may be commented out
DRIVERS_OBJS += fpu.o
DRIVERS_OBJS += gpio.o
DRIVERS_OBJS += uart.o
DRIVERS_OBJS += watchdog.o
DRIVERS_OBJS += led.o
DRIVERS_OBJS += switch.o

APP_OBJS = startup.o handlers.o init.o main.o
APP_OBJS += wdtask.o
APP_OBJS += print.o
APP_OBJS += receive.o
APP_OBJS += lightshow.o
# nostdlib.o must be commented out if standard lib is going to be linked!
APP_OBJS += nostdlib.o


# All object files specified above are prefixed the intermediate directory
OBJS = $(addprefix $(OBJDIR), $(FREERTOS_OBJS) $(FREERTOS_MEMMANG_OBJS) $(FREERTOS_PORT_OBJS) $(DRIVERS_OBJS) $(APP_OBJS))

# Definition of the linker script and final targets
LINKER_SCRIPT = $(addprefix $(APP_SRC), tiva.ld)
ELF_IMAGE = image.elf
TARGET = image.bin

# Include paths to be passed to $(CC) where necessary
INC_FREERTOS = $(FREERTOS_SRC)include/
KOVACIC_INC_DRIVERS = $(KOVACIC_DRIVERS_SRC)include/

# Tivaware headers
TIVAWARE_INC = drivers/tivaware/inc/
TIVAWARE_INC_DRIVERLIB = drivers/tivaware/driverlib/
TIVAWARE_INC_GRLIB = drivers/tivaware/grlib/
TIVAWARE_INC_SENSORLIB = drivers/tivaware/sensorlib/
TIVAWARE_INC_USBLIB = drivers/tivaware/usblib/


# Complete include flags to be passed to $(CC) where necessary
INC_FLAGS = $(INCLUDEFLAG)$(INC_FREERTOS) $(INCLUDEFLAG)$(APP_SRC) $(INCLUDEFLAG)$(FREERTOS_PORT_SRC) $(INCLUDEFLAG)$(KOVACIC_INC_DRIVERS)

# Dependency on HW specific settings
DEP_BSP = $(KOVACIC_INC_DRIVERS)bsp.h

DEP_FRTOS_CONFIG = $(APP_SRC)/FreeRTOSConfig.h
DEP_SETTINGS = $(DEP_FRTOS_CONFIG)


#
# Make rules:
#

all : $(TARGET)

rebuild : clean all

$(TARGET) : $(OBJDIR) $(ELF_IMAGE)
	$(OBJCOPY) -O binary $(word 2,$^) $@

$(OBJDIR) :
	mkdir -p $@

$(ELF_IMAGE) : $(OBJS) $(LINKER_SCRIPT)
	$(LD) -nostdlib -L $(OBJDIR) -T $(LINKER_SCRIPT) $(OBJS) $(OFLAG) $@

debug : _debug_flags all

debug_rebuild : _debug_flags rebuild

_debug_flags :
	$(eval CFLAGS += $(DEB_FLAG))


# FreeRTOS core

$(OBJDIR)queue.o : $(FREERTOS_SRC)queue.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)list.o : $(FREERTOS_SRC)list.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)tasks.o : $(FREERTOS_SRC)tasks.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)timers.o : $(FREERTOS_SRC)timers.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)croutine.o : $(FREERTOS_SRC)croutine.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)event_groups.o : $(FREERTOS_SRC)event_groups.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@


# HW specific part, in FreeRTOS/Source/portable/$(PORT_COMP_TARGET)

$(OBJDIR)port.o : $(FREERTOS_PORT_SRC)port.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@




# Rules for all MemMang implementations are provided
# Only one of these object files must be linked to the final target

$(OBJDIR)heap_1.o : $(FREERTOS_MEMMANG_SRC)heap_1.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)heap_2.o : $(FREERTOS_MEMMANG_SRC)heap_2.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)heap_3.o : $(FREERTOS_MEMMANG_SRC)heap_3.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)heap_4.o : $(FREERTOS_MEMMANG_SRC)heap_4.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)heap_5.o : $(FREERTOS_MEMMANG_SRC)heap_5.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

# Kovacic's Drivers

$(OBJDIR)sysctl.o : $(KOVACIC_DRIVERS_SRC)sysctl.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)systick.o : $(KOVACIC_DRIVERS_SRC)systick.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)nvic.o : $(KOVACIC_DRIVERS_SRC)nvic.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)scb.o : $(KOVACIC_DRIVERS_SRC)scb.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)interrupt.o : $(KOVACIC_DRIVERS_SRC)interrupt.c
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)gpio.o : $(KOVACIC_DRIVERS_SRC)gpio.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)uart.o : $(KOVACIC_DRIVERS_SRC)uart.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)watchdog.o : $(KOVACIC_DRIVERS_SRC)watchdog.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)led.o : $(KOVACIC_DRIVERS_SRC)led.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)switch.o : $(KOVACIC_DRIVERS_SRC)switch.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)fpu.o : $(KOVACIC_DRIVERS_SRC)fpu.c $(DEP_BSP)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@


# Demo application

$(OBJDIR)startup.o : $(APP_SRC)startup.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)handlers.o : $(APP_SRC)handlers.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)init.o : $(APP_SRC)init.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)main.o : $(APP_SRC)main.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)wdtask.o : $(APP_SRC)wdtask.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)print.o : $(APP_SRC)print.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)receive.o : $(APP_SRC)receive.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)lightshow.o : $(APP_SRC)lightshow.c $(DEP_SETTINGS)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)nostdlib.o : $(APP_SRC)nostdlib.c
	$(CC) $(CFLAG) $(CFLAGS) $< $(OFLAG) $@


# Cleanup directives:

clean_obj :
	$(RM) -r $(OBJDIR)

clean_intermediate : clean_obj
	$(RM) *.elf
	$(RM) *.img

clean : clean_intermediate
	$(RM) *.bin

# Rule for flashing
flash:
	sudo lm4flash image.bin

# Short help instructions:

help :
	@echo
	@echo Valid targets:
	@echo - all: builds missing dependencies and creates the target image \'$(IMAGE)\'.
	@echo - rebuild: rebuilds all dependencies and creates the target image \'$(IMAGE)\'.
	@echo - debug: same as \'all\', also includes debugging symbols to \'$(ELF_IMAGE)\'.
	@echo - debug_rebuild: same as \'rebuild\', also includes debugging symbols to \'$(ELF_IMAGE)\'.
	@echo - clean_obj: deletes all object files, only keeps \'$(ELF_IMAGE)\' and \'$(IMAGE)\'.
	@echo - clean_intermediate: deletes all intermediate binaries, only keeps the target image \'$(IMAGE)\'.
	@echo - clean: deletes all intermediate binaries, incl. the target image \'$(IMAGE)\'.
	@echo - help: displays these help instructions.
	@echo


.PHONY :  all rebuild clean clean_intermediate clean_obj debug debug_rebuild _debug_flags help
