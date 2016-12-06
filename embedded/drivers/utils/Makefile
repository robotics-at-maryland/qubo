
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

CFLAGS = $(CPUFLAG) $(WFLAG) $(FPUFLAG)

OBJDIR = gcc/


#
# Make rules:
#

all : $(OBJDIR)uartstdio.o

$(OBJDIR)uartstdio.o : uartstdio.c uartstdio.h
	$(CC) $(CFLAG) $(CFLAGS) $(INCLUDEFLAG) .. $< $(OFLAG) $@
