//*****************************************************************************
//
// Kentec320x240x16_ssd2119_spi.h - Prototypes fpr the Kentec
//                                  BOOSTXL-K350QVG-S1 TFT display drivers with
//                                  an SSD2119 and SPI interface.
//
// Copyright (c) 2012-2016 Texas Instruments Incorporated.  All rights reserved.
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

#ifndef __KENTEC320X240X16_SSD2119_8BIT_H__
#define __KENTEC320X240X16_SSD2119_8BIT_H__

//*****************************************************************************
//
// Prototypes for the globals exported by this driver.
//
//*****************************************************************************
extern void LED_backlight_ON(void);
extern void LED_backlight_OFF(void);
extern void Kentec320x240x16_SSD2119Init(uint32_t ui32SysClock);
extern const tDisplay g_sKentec320x240x16_SSD2119;

#endif // __KENTEC320X240X16_SSD2119_H__
