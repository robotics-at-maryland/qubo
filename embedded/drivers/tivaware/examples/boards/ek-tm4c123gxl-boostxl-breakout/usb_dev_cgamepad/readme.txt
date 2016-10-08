USB HID Composite Gamepad

This example application enables the evaluation board to act as a dual USB
game pad device supported using the Human Interface Device class. The
mapping of the analog pin to gamepad axis and GPIO to button inputs are
listed below.

Analog Pin Mapping:

- Gamepad 1 X  Axis - PE2/AIN1
- Gamepad 1 Y  Axis - PE1/AIN2
- Gamepad 1 Z  Axis - PD3/AIN4
- Gamepad 1 Rx Axis - PD2/AIN5

- Gamepad 2 X  Axis - PD1/AIN6
- Gamepad 2 Y  Axis - PD0/AIN7
- Gamepad 2 Z  Axis - PE5/AIN8
- Gamepad 2 Ry Axis - PB5/AIN11

Button Pin Mapping.

- Gamepad 1 Button  1 - PF4
- Gamepad 1 Button  2 - PE0
- Gamepad 1 Button  3 - PE3
- Gamepad 1 Button  4 - PE4
- Gamepad 1 Button  5 - PB4
- Gamepad 1 Button  6 - PB3
- Gamepad 1 Button  7 - PB2
- Gamepad 1 Button  8 - PB0
- Gamepad 1 Button  9 - PB1
- Gamepad 1 Button 10 - PA6
- Gamepad 1 Button 11 - PA7

- Gamepad 2 Button  1 - PF0
- Gamepad 2 Button  2 - PC4
- Gamepad 2 Button  3 - PC5
- Gamepad 2 Button  4 - PC6
- Gamepad 2 Button  5 - PC7
- Gamepad 2 Button  6 - PD6
- Gamepad 2 Button  7 - PD7
- Gamepad 2 Button  8 - PA5
- Gamepad 2 Button  9 - PA4
- Gamepad 2 Button 10 - PA3
- Gamepad 2 Button 11 - PA2

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

This is part of revision 2.1.3.156 of the EK-TM4C123GXL Firmware Package.
