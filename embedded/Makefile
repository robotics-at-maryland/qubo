# Ross Baehr
# R@M 2017
# ross.baehr@gmail.com

# FreeRTOS and src objects are put into obj/
# Tivaware objects are built in drivers/ and symlinked into obj/

LINKER_SCRIPT = $(SRC)tiva.ld
ELF_IMAGE = image.elf
TARGET = image.bin

# You shouldn't need to edit anything below here
#########################################################################
# FLAGS                                                                 #
#########################################################################

TOOLCHAIN = arm-none-eabi-
CC = $(TOOLCHAIN)gcc
CXX = $(TOOLCHAIN)g++
AS = $(TOOLCHAIN)as
LD = $(TOOLCHAIN)ld
OBJCOPY = $(TOOLCHAIN)objcopy
AR = $(TOOLCHAIN)ar

# GCC flags
#
CFLAG = -c
OFLAG = -o
INCLUDEFLAG = -I

CFLAGS = -g -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS +=-Os -ffunction-sections -fdata-sections -MD -std=c99
CFLAGS += -pedantic -DPART_TM4C123GH6PM -c
CFLAGS += -DTARGET_IS_TM4C123_RB2
CFLAGS += -Dgcc
#CFLAGS = -mcpu=cortex-m4 -march=armv7e-m -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -Dgcc -DPART_TM4C123GH6PM -DTARGET_IS_TM4C123_RB1 -ffunction-sections -fdata-sections -g -gdwarf-3 -gstrict-dwarf -specs="nosys.specs" -MD -std=c99 -I$(TC_PATH)arm-none-eabi/include/

LDFLAGS = -T $(LINKER_SCRIPT) --entry ResetISR --gc-sections

# Default debug off unless ran with make debug
DEBUG = OFF

##########################################################################
# LOCATIONS                                                              #
##########################################################################

# Directory with HW drivers' source files
DRIVERS_SRC = drivers/

# Directory with demo specific source (and header) files
SRC = src/

# Subdir of src
TASKDIR = tasks/

# subdir of src for interrupts
INTERRUPTS = interrupts/

# Intermediate directory for all *.o and other files:
OBJDIR = obj/

# These objects are needed for linking
LIBGCC = $(shell $(CC) $(CFLAGS) -print-libgcc-file-name)
LIBC = $(shell $(CC) $(CFLAGS) -print-file-name=libc.a)
LIBM = $(shell $(CC) $(CFLAGS) -print-file-name=libm.a)

# Additional C compiler flags to produce debugging symbols
DEB_FLAG = -g -DDEBUG

# Compiler/target path in FreeRTOS/Source/portable
PORT_COMP_TARG = GCC/ARM_CM4F/

# FreeRTOS source base directory
FREERTOS_SRC = FreeRTOS/Source/

# Directory with memory management source files
FREERTOS_MEMMANG_SRC = $(FREERTOS_SRC)portable/MemMang/

# Directory with platform specific source files
FREERTOS_PORT_SRC = $(FREERTOS_SRC)portable/$(PORT_COMP_TARG)

# Qubobus src files
QUBOBUS_SRC = qubobus/src/

##########################################################################
# OBJECTS                                                                #
##########################################################################

# Object files for FreeRTOS
FREERTOS_OBJS = queue.o list.o tasks.o stream_buffer.o
# The following o. files are only necessary if
# certain options are enabled in FreeRTOSConfig.h
#FREERTOS_OBJS += timers.o
#FREERTOS_OBJS += croutine.o
#FREERTOS_OBJS += event_groups.o

# Only one memory management .o file must be uncommented!
#FREERTOS_MEMMANG_OBJS = heap_1.o
#FREERTOS_MEMMANG_OBJS = heap_2.o
#FREERTOS_MEMMANG_OBJS = heap_3.o
FREERTOS_MEMMANG_OBJS = heap_4.o
#FREERTOS_MEMMANG_OBJS = heap_5.o

 FREERTOS_PORT_OBJS = port.o

# Driver object lists to tivaware libs
DRIVERLIB_OBJS := $(wildcard $(OBJDIR)driverlib/*.o)
DRIVERLIB_OBJS := $(DRIVERLIB_OBJS) $(wildcard $(OBJDIR)driverlib/*.a)
UTILS_OBJS = $(wildcard $(OBJDIR)utils/*.o)

USBLIB_OBJS = $(wildcard $(OBJDIR)usblib/*.o)

TIVA_DRIVER_OBJS = $(DRIVERLIB_OBJS)

# List of source file objects. Adds any c file in $(SRC) and $(SRC)$(TASKDIR)
SRC_C_FILES = $(wildcard $(SRC)*.c) $(wildcard $(SRC)$(TASKDIR)*.c) $(wildcard $(SRC)lib/*.c) $(wildcard $(SRC)$(INTERRUPTS)*.c)
SRC_OBJS := $(SRC_C_FILES:$(SRC)%=%)
SRC_OBJS := $(SRC_OBJS:.c=.o)


QUBOBUS_OBJECTS = io.o protocol.o embedded.o safety.o battery.o power.o thruster.o pneumatics.o depth.o debug.o

# All object files specified above are prefixed the object directory
OBJS = $(addprefix $(OBJDIR), $(FREERTOS_OBJS) $(FREERTOS_MEMMANG_OBJS) $(FREERTOS_PORT_OBJS) \
	$(SRC_OBJS) $(QUBOBUS_OBJECTS))
OBJS += $(USBLIB_OBJS)
OBJS += $(DRIVERLIB_OBJS)

##########################################################################
# INCLUDES                                                               #
##########################################################################
# FreeRTOS core include files
INC_FREERTOS = $(FREERTOS_SRC)include/

INC_QUBOBUS = qubobus/include/

# Complete include flags to be passed to $(CC) where necessary
INC_FLAGS = $(addprefix $(INCLUDEFLAG), $(INC_FREERTOS) $(FREERTOS_PORT_SRC) $(DRIVERS_SRC) $(SRC) $(INC_QUBOBUS))

DEP_FRTOS_CONFIG = $(SRC)FreeRTOSConfig.h

##########################################################################
# RULES                                                                  #
##########################################################################

all : $(TARGET)

debug: clean debug_flag all

debug_flag: tiva
	$(eval DEBUG := ON)
	$(eval CFLAGS += $(DEB_FLAG))

tiva:
	bash -c "cd drivers;./symlink_objs &> /dev/null"

rebuild : clean

setenv:
	source setenv.sh

dbgrun:
	./start_openocd.sh

dbg:
	arm-none-eabi-gdb $(ELF_IMAGE)

$(TARGET) : $(OBJDIR) $(ELF_IMAGE)
	$(OBJCOPY) -O binary $(word 2,$^) $@

$(OBJDIR) :
	mkdir -p $@ $(OBJDIR)$(TASKDIR) $(OBJDIR)lib/ $(OBJDIR)$(INTERRUPTS)

$(ELF_IMAGE) : $(OBJS) $(LIBC) $(LIBM) $(LIBGCC)
	@if [ $(DEBUG) = "ON" ]; then \
		echo "_____________________________________________________________";\
		echo "Debug on";\
		echo "$(LD) $(OFLAG) $@ $^ $(UTILS_OBJS) $(LDFLAGS)";\
		$(LD) $(OFLAG) $@ $^ $(UTILS_OBJS) $(LDFLAGS);\
	else \
		echo "_____________________________________________________________";\
		echo "Debug off";\
		echo "$(LD) $(OFLAG) $@ $^ $(LDFLAGS)";\
		$(LD) $(OFLAG) $@ $^ $(LDFLAGS);\
	fi

# Objects for files in $(SRC) and $(SRC)$(TASKDIR)
$(OBJDIR)%.o: $(SRC)%.c $(SRC)include/* $(OBJDIR)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

# Make objects for the FreeRTOS Files
$(OBJDIR)%.o: $(FREERTOS_SRC)%.c $(INC_FREERTOS)* $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)%.o: $(FREERTOS_MEMMANG_SRC)%.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)%.o: $(FREERTOS_PORT_SRC)%.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)%.o: $(QUBOBUS_SRC)%.c $(INC_QUBOBUS)*
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@


# Cleanup directives:

clean_obj :
	$(RM) $(OBJDIR)*.o $(OBJDIR)*.d
	$(RM) $(OBJDIR)$(TASKDIR)*.o $(OBJDIR)$(TASKDIR)*.d
	$(RM) $(OBJDIR)lib/*.o	$(RM) $(OBJDIR)lib/*.d
	$(RM) $(OBJDIR)$(INTERRUPTS)*.o $(OBJDIR)$(INTERRUPTS)*.d


clean_intermediate : clean_obj
	$(RM) *.elf
	$(RM) *.img

clean : clean_intermediate
	$(RM) *.bin

# Rule for flashing
flash:
	sudo /opt/lm4tools/lm4flash/lm4flash ./image.bin

# Short help instructions:

print-%  : ; @echo $* = $($*)

.PHONY :  all rebuild clean clean_intermediate clean_obj debug debug_rebuild _debug_flags help dbgrun dbg setenv
