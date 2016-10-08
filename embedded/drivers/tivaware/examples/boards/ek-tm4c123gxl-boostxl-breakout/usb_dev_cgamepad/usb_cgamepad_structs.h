//*****************************************************************************
//
// usb_keyb_structs.h - Data structures defining the game pad USB device.
//
// Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the EK-TM4C123GXL Firmware Package.
//
//*****************************************************************************

#ifndef _USB_GAMEPAD_STRUCTS_H_
#define _USB_GAMEPAD_STRUCTS_H_

//****************************************************************************
//
// The number of individual device class instances comprising this composite
// device.
//
//****************************************************************************
#define NUM_DEVICES             2

extern tCompositeEntry g_psCompDevices[NUM_DEVICES];

extern uint32_t GamepadHandler(void *pvCBData, uint32_t ui32Event,
                               uint32_t ui32MsgData, void *pvMsgData);

extern tUSBDHIDGamepadDevice g_sGamepadDeviceA;
extern tUSBDHIDGamepadDevice g_sGamepadDeviceB;
extern tUSBDCompositeDevice g_sCompGameDevice;

#endif
