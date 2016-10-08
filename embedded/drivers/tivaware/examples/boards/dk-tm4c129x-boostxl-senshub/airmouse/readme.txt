Motion Air Mouse

This example demonstrates the use of the Sensor Library, DK-TM4C129X and
the SensHub BoosterPack to fuse nine axis sensor measurements into motion
and gesture events.  These events are then transformed into mouse and
keyboard events to perform standard HID tasks.

Connect the USB OTG port, between the EM connectors of the DK-TM4C129X, to
a standard computer USB port.  The DK-TM4C129X with SensHub BoosterPack
enumerates on the USB bus as a composite HID keyboard and mouse.

Hold the DK-TM4C129X with the buttons and LCD away from the user and toward
the computer with USB Device cable exiting from the left side of the board.

- Roll or tilt the DK-TM4C129X to move the mouse cursor of the computer
up, down, left and right.

- The buttons on the DK-TM4C129X, SEL and DOWN, perform the left and right
mouse click actions respectively.  The buttons on the SensHub BoosterPack
are not currently used by this example.

- A quick spin of the DK-TM4C129X generates a PAGE_UP or PAGE_DOWN keyboard
press and release depending on the direction of the spin.  This motion
simulates scrolling.

- A quick horizontal jerk to the left or right  generates a CTRL+ or CTRL-
keyboard event, which creates the zoom effect used in many applications,
especially web browsers.

- A quick vertical lift generates an ALT+TAB keyboard event, which allows
the computer user to select between currently open windows.

- A quick twist to the left or right moves the window selector.

- A quick jerk in the down direction selects the desired window and
closes the window selection dialog.

This example also supports the RemoTI low power RF Zigbee&reg;&nbsp;human
interface device profile.  The wireless features of this example require
the CC2533EMK expansion card and the CC2531EMK USB Dongle.  For details and
instructions for wireless operations see the Wiki at
http://processors.wiki.ti.com/index.php/Wireless_Air_Mouse_Guide.

-------------------------------------------------------------------------------

Copyright (c) 2014-2016 Texas Instruments Incorporated.  All rights reserved.
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

This is part of revision 2.1.3.156 of the DK-TM4C129X Firmware Package.
