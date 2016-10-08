Pressure Measurement with the BMP180

This example demonstrates the basic use of the Sensor Library, the
EK-TM4C1294XL LaunchPad, and the SensHub BoosterPack to obtain air pressure
and temperature measurements with the BMP180 sensor.

SensHub BoosterPack (BOOSTXL-SENSHUB) must be installed on BoosterPack 1
interface headers.

Instructions for use of SensorHub on BoosterPack 2 headers are in the code
comments.

Connect a serial terminal program to the LaunchPad's ICDI virtual serial
port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
The raw sensor measurements are printed to the terminal.  The LED
blinks at 1 Hz once the initialization is complete and the example is
running.

-------------------------------------------------------------------------------

Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
Software License Agreement

Texas Instruments (TI) is supplying this software for use solely and
exclusively on TI's microcontroller products. The software is owned by
TI and/or its suppliers, and is protected under applicable copyright
laws. You may not combine this software with "viral" open-source
software in order to form a larger program.

THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, FOR ANY REASON WHATSOEVER.

This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
