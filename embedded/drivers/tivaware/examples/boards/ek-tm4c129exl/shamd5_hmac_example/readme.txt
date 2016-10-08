SHA/MD5 HMAC Demo

This application demonstrates the use of HMAC operation with the available
algorithms of the SHA/MD5 module like MD5, SHA1, SHA224 and SHA256.

This application uses a command-line based interface through a virtual COM
port on UART 0, with the settings 115,200-8-N-1.

Using the command prompt, user can configure the SHA/MD5 module to select
an algorithm to perform HMAC operation.  User can also enter key and data
values during runtime.  Type "help" on the terminal, once the prompt is
displayed, for details of these configuration.

The following web site has been used to validate the HMAC output for these
algorithms: http://www.freeformatter.com/hmac-generator.html

Please note that uDMA is not used in this example.

-------------------------------------------------------------------------------

Copyright (c) 2015-2016 Texas Instruments Incorporated.  All rights reserved.
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

This is part of revision 2.1.3.156 of the EK-TM4C129EXL Firmware Package.
