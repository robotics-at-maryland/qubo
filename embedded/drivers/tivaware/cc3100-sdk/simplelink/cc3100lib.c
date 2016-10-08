//*****************************************************************************
//
// cc3100lib.c - CC3100 library includes.
//
// Copyright (c) 2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
// 
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
// 
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************

#ifndef __CC3100LIB_C__
#define __CC3100LIB_C__

//*****************************************************************************
//
// If building with a C++ compiler, make all of the definitions in this header
// have a C binding.
//
//*****************************************************************************
#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdbool.h>

//*****************************************************************************
//
// Include Simple Link header files from "include" folder.
//
//*****************************************************************************
#include "cc3100-sdk/simplelink/include/device.h"
#include "cc3100-sdk/simplelink/include/fs.h"
#include "cc3100-sdk/simplelink/include/netapp.h"
#include "cc3100-sdk/simplelink/include/netcfg.h"
#include "cc3100-sdk/simplelink/include/simplelink.h"
#include "cc3100-sdk/simplelink/include/socket.h"
#include "cc3100-sdk/simplelink/include/trace.h"
#include "cc3100-sdk/simplelink/include/wlan.h"
#include "cc3100-sdk/simplelink/include/wlan_rx_filters.h"

//*****************************************************************************
//
// Include Simple Link header files from "source" folder.
//
//*****************************************************************************
#include "cc3100-sdk/simplelink/source/protocol.h"
#include "cc3100-sdk/simplelink/source/driver.h"
#include "cc3100-sdk/simplelink/source/flowcont.h"
#include "cc3100-sdk/simplelink/source/nonos.h"
#include "cc3100-sdk/simplelink/source/objInclusion.h"
#include "cc3100-sdk/simplelink/source/spawn.h"

//*****************************************************************************
//
// Include Simple Link source code from "source" folder.
//
//*****************************************************************************
#include "cc3100-sdk/simplelink/source/device.c"
#include "cc3100-sdk/simplelink/source/driver.c"
#include "cc3100-sdk/simplelink/source/flowcont.c"
#include "cc3100-sdk/simplelink/source/fs.c"
#include "cc3100-sdk/simplelink/source/netapp.c"
#include "cc3100-sdk/simplelink/source/netcfg.c"
#include "cc3100-sdk/simplelink/source/nonos.c"
#include "cc3100-sdk/simplelink/source/socket.c"
#include "cc3100-sdk/simplelink/source/spawn.c"
#include "cc3100-sdk/simplelink/source/wlan.c"

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif
#endif // __CC3100LIB_C__
