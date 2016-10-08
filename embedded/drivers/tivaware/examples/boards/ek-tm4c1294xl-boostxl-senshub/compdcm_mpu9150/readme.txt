Nine Axis Sensor Fusion with the MPU9150 and Complimentary-Filtered
DCM

This example demonstrates the basic use of the Sensor Library, TM4C1294
LaunchPad and SensHub BoosterPack to obtain nine axis motion measurements
from the MPU9150.  The example fuses the nine axis measurements into a set
of Euler angles: roll, pitch and yaw.  It also produces the rotation
quaternions.  The fusion mechanism demonstrated is a complimentary-filtered
direct cosine matrix (DCM) algorithm. The algorithm is provided as part of
the Sensor Library.

This example requires that the BOOSTXL-SENSHUB be installed on BoosterPack
1 interface headers.  See code comments for instructions on how to use
BoosterPack 2 interface.

Connect a serial terminal program to the LaunchPad's ICDI virtual serial
port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
The raw sensor measurements, Euler angles and quaternions are printed to
the terminal.  An LED begins to blink at 1Hz after initialization is
completed and the example application is running.

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
