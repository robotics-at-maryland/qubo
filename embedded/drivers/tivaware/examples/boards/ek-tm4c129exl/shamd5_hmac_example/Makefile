#******************************************************************************
#
# Makefile - Rules for building the SHA/MD5 HMAC Demo.
#
# Copyright (c) 2015-2016 Texas Instruments Incorporated.  All rights reserved.
# Software License Agreement
# 
# Texas Instruments (TI) is supplying this software for use solely and
# exclusively on TI's microcontroller products. The software is owned by
# TI and/or its suppliers, and is protected under applicable copyright
# laws. You may not combine this software with "viral" open-source
# software in order to form a larger program.
# 
# THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
# NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
# NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
# CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
# DAMAGES, FOR ANY REASON WHATSOEVER.
# 
# This is part of revision 2.1.3.156 of the EK-TM4C129EXL Firmware Package.
#
#******************************************************************************

#
# Defines the part type that this project uses.
#
PART=TM4C129ENCPDT

#
# The base directory for TivaWare.
#
ROOT=../../../..

#
# Include the common make definitions.
#
include ${ROOT}/makedefs

#
# Where to find source files that do not live in this directory.
#
VPATH=../drivers
VPATH+=../../../../utils

#
# Where to find header files that do not live in the source directory.
#
IPATH=..
IPATH+=../../../..

#
# The default rule, which causes the SHA/MD5 HMAC Demo to be built.
#
all: ${COMPILER}
all: ${COMPILER}/shamd5_hmac_example.axf

#
# The rule to clean out all the build products.
#
clean:
	@rm -rf ${COMPILER} ${wildcard *~}

#
# The rule to create the target directory.
#
${COMPILER}:
	@mkdir -p ${COMPILER}

#
# Rules for building the SHA/MD5 HMAC Demo.
#
${COMPILER}/shamd5_hmac_example.axf: ${COMPILER}/cmdline.o
${COMPILER}/shamd5_hmac_example.axf: ${COMPILER}/pinout.o
${COMPILER}/shamd5_hmac_example.axf: ${COMPILER}/shamd5_hmac_example.o
${COMPILER}/shamd5_hmac_example.axf: ${COMPILER}/startup_${COMPILER}.o
${COMPILER}/shamd5_hmac_example.axf: ${COMPILER}/uartstdio.o
${COMPILER}/shamd5_hmac_example.axf: ${COMPILER}/ustdlib.o
${COMPILER}/shamd5_hmac_example.axf: ${ROOT}/driverlib/${COMPILER}/libdriver.a
${COMPILER}/shamd5_hmac_example.axf: shamd5_hmac_example.ld
SCATTERgcc_shamd5_hmac_example=shamd5_hmac_example.ld
ENTRY_shamd5_hmac_example=ResetISR
CFLAGSgcc=-DTARGET_IS_TM4C129_RA2

#
# Include the automatically generated dependency files.
#
ifneq (${MAKECMDGOALS},clean)
-include ${wildcard ${COMPILER}/*.d} __dummy__
endif
