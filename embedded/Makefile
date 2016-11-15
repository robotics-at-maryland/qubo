# If you have questions about this or something doesn't work, ask Ross
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
CFLAG = -c
OFLAG = -o
INCLUDEFLAG = -I
CPUFLAG = -mthumb -mcpu=cortex-m4 -DTARGET_IS_TM4C123_RB1
WFLAG = -Wall -Wextra #-Werror
FPUFLAG=-mfpu=fpv4-sp-d16 -mfloat-abi=softfp

CFLAGS = $(CPUFLAG) $(WFLAG) $(FPUFLAG)

##########################################################################
# LOCATIONS                                                              #
##########################################################################

# Directory with HW drivers' source files
DRIVERS_SRC = drivers/

# Directory with demo specific source (and header) files
SRC = src/

# Subdir of src
TASKDIR = tasks/

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

##########################################################################
# OBJECTS                                                                #
##########################################################################

# Object files for FreeRTOS
FREERTOS_OBJS = queue.o list.o tasks.o
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

TIVA_DRIVER_OBJS = $(DRIVERLIB_OBJS) $(UTILS_OBJS)

# List of source file objects. Adds any c file in $(SRC) and $(SRC)$(TASKDIR)
SRC_C_FILES = $(wildcard $(SRC)*.c) $(wildcard $(SRC)$(TASKDIR)*.c)
SRC_OBJS := $(SRC_C_FILES:$(SRC)%=%)
SRC_OBJS := $(SRC_OBJS:.c=.o)

# All object files specified above are prefixed the object directory
OBJS = $(addprefix $(OBJDIR), $(FREERTOS_OBJS) $(FREERTOS_MEMMANG_OBJS) $(FREERTOS_PORT_OBJS) \
	$(SRC_OBJS))

##########################################################################
# INCLUDES                                                               #
##########################################################################
# FreeRTOS core include files
INC_FREERTOS = $(FREERTOS_SRC)include/

# Complete include flags to be passed to $(CC) where necessary
INC_FLAGS = $(addprefix $(INCLUDEFLAG), $(INC_FREERTOS) $(FREERTOS_PORT_SRC) $(DRIVERS_SRC) $(SRC))

DEP_FRTOS_CONFIG = $(SRC)FreeRTOSConfig.h

##########################################################################
# RULES                                                                  #
##########################################################################

all : $(TARGET)

rebuild : clean all

$(TARGET) : $(OBJDIR) $(ELF_IMAGE)
	$(OBJCOPY) -O binary $(word 2,$^) $@

tiva:
	bash -c "make -C drivers/"

$(OBJDIR) :
	mkdir -p $@

$(ELF_IMAGE) : $(OBJS) $(LINKER_SCRIPT)
	$(LD) -L $(OBJDIR) -T $(LINKER_SCRIPT) $(TIVA_DRIVER_OBJS) $(OBJS) $(LIBGCC) $(LIBC) $(LIBM) $(OFLAG) $@

debug : _debug_flags all

debug_rebuild : _debug_flags rebuild

_debug_flags :
	$(eval CFLAGS += $(DEB_FLAG))

$(TASKDIR) :
	mkdir -p $(OBJDIR)$(TASKDIR)


# Objects for files in $(SRC) and $(SRC)$(TASKDIR)
$(OBJDIR)%.o: $(SRC)%.c $(wildcard $(SRC)%.h) $(TASKDIR)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

#$(OBJDIR)$(TASKDIR)%.o: $(TASKDIR) $(SRC)$(TASKDIR)%.c $(wildcard $(SRC)$(TASKDIR)%.h)
#	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@


# Make objects for the FreeRTOS Files
$(OBJDIR)%.o: $(FREERTOS_SRC)%.c $(INC_FREERTOS)* $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)%.o: $(FREERTOS_MEMMANG_SRC)%.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

$(OBJDIR)%.o: $(FREERTOS_PORT_SRC)%.c $(DEP_FRTOS_CONFIG)
	$(CC) $(CFLAG) $(CFLAGS) $(INC_FLAGS) $< $(OFLAG) $@

# Cleanup directives:

clean_obj :
	$(RM) $(OBJDIR)*
	$(RM) $(OBJDIR)$(TASKDIR)*

clean_intermediate : clean_obj
	$(RM) *.elf
	$(RM) *.img

clean : clean_intermediate
	$(RM) *.bin

# Rule for flashing
flash:
	sudo lm4flash image.bin

# Short help instructions:

print-%  : ; @echo $* = $($*)

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
