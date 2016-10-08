Fuel Tank BoosterPack Measurement Example Application


This example demonstrates the basic use of the Sensor Library, TM4C1294XL
LaunchPad and the Fuel Tank BoosterPack to obtain state-of-charge,
battery voltage, temperature, and several other supported  measurements
via the BQ27510G3 gas gauge sensor on the Fuel tank boosterpack.

The LEDs on the LaunchPad will blink while the application is running.

The Fuel Tank BoosterPack (BOOSTXL-BATTPACK) defaults to be installed on
the BoosterPack 2 interface headers.

Instructions for use of Fuel Tank on BoosterPack 1 headers are in the code
comments.

If you would like to observe how the application affects the voltage or
current readings from the battery, please ensure the POWER_SELECT (JP1)
jumper on the EK-TM4C1294XL LaunchPad is configured for "BoosterPack".

Connect a serial terminal program to the LaunchPad's ICDI virtual serial
port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
The raw sensor measurements are printed to the terminal.

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
