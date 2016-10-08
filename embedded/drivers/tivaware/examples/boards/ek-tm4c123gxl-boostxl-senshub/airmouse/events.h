//*****************************************************************************
//
// events.h - Events that control software tasks.
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

#ifndef __EVENTS_H__
#define __EVENTS_H__

//*****************************************************************************
//
// Number of SysTick Timer interrupts per second.
//
//*****************************************************************************
#define SYSTICKS_PER_SECOND     100

//*****************************************************************************
//
// Global system tick counter.  incremented by SysTickIntHandler.
//
//*****************************************************************************
extern volatile uint_fast32_t g_ui32SysTickCount;

//*****************************************************************************
//
// Hold the state of the buttons on the board.
//
//*****************************************************************************
extern volatile uint_fast8_t g_ui8Buttons;

//*****************************************************************************
//
// Holds command bits used to signal the main loop to perform various tasks.
//
//*****************************************************************************
extern volatile uint_fast32_t g_ui32Events;

#define USB_TICK_EVENT          0
#define MOTION_EVENT            1
#define MOTION_ERROR_EVENT      2
#define LPRF_EVENT              3
#define LPRF_TICK_EVENT         4

#endif // __EVENTS_H__
