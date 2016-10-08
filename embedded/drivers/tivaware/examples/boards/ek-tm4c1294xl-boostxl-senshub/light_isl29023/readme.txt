Light Measurement with the ISL29023

This example demonstrates the basic use of the Sensor Library, TM4C1294
Connected LaunchPad and the SensHub BoosterPack to obtain ambient and
infrared light measurements with the ISL29023 sensor.

The SensHub BoosterPack must be installed on BoosterPack 1 interface.
See code comments for changes needed to use BoosterPack 2 interface.

Connect a serial terminal program to the LaunchPad's ICDI virtual serial
port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
The raw sensor measurements are printed to the terminal.  An LED blinks at
1Hz once the initialization is complete and the example is running.

The code automatically adjusts the dynamic range of the sensor when the
intensity reaches a min or max threshold within the current range setting.


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
