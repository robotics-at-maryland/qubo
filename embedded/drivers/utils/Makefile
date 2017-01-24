
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
FPUFLAG=-mfpu=fpv4-sp-d16 -mfloat-abi=hard

CFLAGS = $(CPUFLAG) $(WFLAG) $(FPUFLAG)

OBJDIR = gcc/

ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/../

INCLUDE = $(addprefix $(INCLUDEFLAG), $(ROOT_DIR))


#
# Make rules:
#

all : $(OBJDIR)uartstdio.o $(OBJDIR)softi2c.o

$(OBJDIR)uartstdio.o : uartstdio.c uartstdio.h
	$(CC) $(CFLAG) $(CFLAGS) $(INCLUDE) $< $(OFLAG) $@

$(OBJDIR)softi2c.o : softi2c.c softi2c.h
	$(CC) $(CFLAG) $(CFLAGS) $(INCLUDE) $< $(OFLAG) $@

print-% : ; @echo $* = $($*)
