# Tiva Makefile
# #####################################
#
# Part of the uCtools project
# uctools.github.com
#
#######################################
# user configuration:
#######################################
#######################################
# user configuration:
#######################################
# TARGET: name of the output file
TARGET = main
# MCU: part number to build for
MCU = TM4C123GH6PM
# TIVAWARE_PATH: path to tivaware folder
TIVAWARE_PATH = $(HOME)/robotics/tivaware
# SOURCES: list of input source sources
SOURCES = main.c startup_gcc.c sysctl.c  i2c.c gpio.c uartstdio.c uart.c ustdlib.c bmp180.c
#$(TIVAWARE_PATH)/driverlib/*.c $(TIVAWARE_PATH)/inc/*.c
# INCLUDES: list of includes, by default, use Includes directory
INCLUDES = -IInclude
# OUTDIR: directory to use for output
OUTDIR = build


# LD_SCRIPT: linker script
LD_SCRIPT = $(MCU).ld

# define flags
CFLAGS = -g -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=softfp
CFLAGS += -O0 -ffunction-sections -fdata-sections -MD -std=c99 -Wall
CFLAGS += -pedantic -DPART_$(MCU) -c -I$(TIVAWARE_PATH)
CFLAGS += -DTARGET_IS_BLIZZARD_RA1
LDFLAGS = -T $(LD_SCRIPT) --entry ResetISR --gc-sections

#######################################
# end of user configuration
#######################################
#
#######################################
# binaries
#######################################
CC = arm-none-eabi-gcc
LD = arm-none-eabi-ld
OBJCOPY = arm-none-eabi-objcopy
RM      = rm -f
MKDIR	= mkdir -p
#######################################

# list of object files, placed in the build directory regardless of source path
OBJECTS = $(addprefix $(OUTDIR)/,$(notdir $(SOURCES:.c=.o)))

# default: build bin
all: $(OUTDIR)/$(TARGET).bin

$(OUTDIR)/%.o: src/%.c | $(OUTDIR)
	$(CC) -o $@ $^ $(CFLAGS)


#$(TIVAWARE_PATH)/driverlib/%.o: $(TIVAWARE_PATH)/driverlib/%.c | $(OUTDIR)
#	$(CC) -o $@ $^ $(CFLAGS) 

#$(TIVAWARE_PATH)/inc/%.o: $(TIVAWARE_PATH)/inc/%.c | $(OUTDIR)
#	$(CC) -o $@ $^ $(CFLAGS) 

$(OUTDIR)/a.out: $(OBJECTS)
	$(LD) -o $@ $^ $(LDFLAGS)

$(OUTDIR)/main.bin: $(OUTDIR)/a.out
	$(OBJCOPY) -O binary $< $@

flash: $(OUTDIR)/main.bin
	sudo lm4flash $(OUTDIR)/$(TARGET).bin

# create the output directory
$(OUTDIR):
	$(MKDIR) $(OUTDIR)

clean:
	-$(RM) $(OUTDIR)/*

.PHONY: all clean flash
