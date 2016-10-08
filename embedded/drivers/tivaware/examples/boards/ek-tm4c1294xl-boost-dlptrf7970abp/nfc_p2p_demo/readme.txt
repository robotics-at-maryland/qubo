NFC P2P Demo

This example application demonstrates the operation of the Tiva C Series
evaluation kit with the TRF7970ABP BoosterPack as a NFC P2P device.

The application supports reading and writing Text, URI, and SmartPoster
Tags. The application gets a raw message buffer from the TRF79x0 stack,
decodes the information to recognized tag types, then re-encodes the data
to a buffer to be sent back out. Pressing switch SW1 sends a URI message
with a link to the Tiva C series Launchpad website. Pressing switch SW2
echo's back the last tag recieved. If no tag has been recieved then this
button does nothing. Full debug information is given across the UART0
channel to aid in NFC P2P development.

This application assumes the TRF7970ABP is connected to the boosterpack 2
headers on the development kit. To use the boosterpack 1 headers you will
need to toggle the TRF79X0_USE_BOOSTERPACK_2 define in trf79x0_hw.h
and recompile the application.

For more information on NFC please see the full NFC specification list at
http://www.nfc-forum.org/specs/spec_list/ .

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
