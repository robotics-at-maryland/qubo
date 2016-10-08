Capacitive Touch Example

An example that works with the 430BOOST-SENSE1 capactive sense
BoosterPack, originally designed for the MSP430 LaunchPad.

The TM4C123GH6PM does not have the capacitive sense hardware assisted
peripheral features of some MSP430 chips.  Therefore it is required that
the user install surface mount resistors on the pads provided on the bottom
of the capacitive sense BoosterPack.  Resistor values of 200k ohms are are
recommended.  Calibration may be required even when using 200k ohm
resistors as each capsense booster pack varies.  Calibration is required
for resistors other than 200k ohm.

See the wiki page for calibration
procedure.  http://processors.wiki.ti.com/index.php/tm4c123g-launchpad

-------------------------------------------------------------------------------

Copyright (c) 2011-2016 Texas Instruments Incorporated.  All rights reserved.
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

This is part of revision 2.1.3.156 of the EK-TM4C123GXL Firmware Package.
