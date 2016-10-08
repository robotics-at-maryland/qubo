#******************************************************************************
#
# Makefile - Rules for building the SensorHub Internet of Things example.
#
# Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
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
# This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
#
#******************************************************************************

#
# Defines the part type that this project uses.
#
PART=TM4C1294NCPDT

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
VPATH=../../../../third_party/FreeRTOS/Source/portable/GCC/ARM_CM4F
VPATH+=../../../../third_party/FreeRTOS/Source/portable/MemMang/
VPATH+=../../../../third_party/FreeRTOS/Source
VPATH+=../drivers
VPATH+=../../../../third_party/exosite
VPATH+=../../../../utils

#
# Where to find header files that do not live in the source directory.
#
IPATH=.
IPATH+=..
IPATH+=../../../..
IPATH+=../../../../third_party/FreeRTOS/Source/portable/GCC/ARM_CM4F
IPATH+=../../../../third_party/FreeRTOS
IPATH+=../../../../third_party/lwip-1.4.1/apps
IPATH+=../../../../third_party/exosite
IPATH+=../../../../third_party/FreeRTOS/Source/include
IPATH+=../../../../third_party/lwip-1.4.1/ports/tiva-tm4c129/include
IPATH+=../../../../third_party/lwip-1.4.1/src/include
IPATH+=../../../../third_party/lwip-1.4.1/src/include/ipv4
IPATH+=../../../../third_party
IPATH+=../../../../third_party

#
# The default rule, which causes the SensorHub Internet of Things example to be built.
#
all: ${COMPILER}
all: ${COMPILER}/senshub_iot.axf

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
# Rules for building the SensorHub Internet of Things example.
#
${COMPILER}/senshub_iot.axf: ${COMPILER}/bmp180_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/buttons.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/cloud_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/cmdline.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/command_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/compdcm_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/eth_client_lwip.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/exosite.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/exosite_hal_lwip.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/exosite_meta.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/flash_pb.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/heap_1.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/http.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/isl29023_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/list.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/locator.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/lwiplib.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/pinout.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/port.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/queue.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/ringbuf.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/senshub_iot.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/sht21_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/startup_${COMPILER}.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/tasks.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/timers.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/tmp006_task.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/uartstdio.o
${COMPILER}/senshub_iot.axf: ${COMPILER}/ustdlib.o
${COMPILER}/senshub_iot.axf: ${ROOT}/sensorlib/${COMPILER}/libsensor.a
${COMPILER}/senshub_iot.axf: ${ROOT}/driverlib/${COMPILER}/libdriver.a
${COMPILER}/senshub_iot.axf: senshub_iot.ld
SCATTERgcc_senshub_iot=senshub_iot.ld
ENTRY_senshub_iot=ResetISR
CFLAGSgcc=-DTARGET_IS_TM4C129_RA0 -DUART_BUFFERED

#
# Include the automatically generated dependency files.
#
ifneq (${MAKECMDGOALS},clean)
-include ${wildcard ${COMPILER}/*.d} __dummy__
endif
