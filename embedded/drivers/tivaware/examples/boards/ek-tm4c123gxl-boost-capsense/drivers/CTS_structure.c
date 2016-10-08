//*****************************************************************************
//
// CTS_structure.c - Capacative Sense structures.
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

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"
#include "inc/hw_memmap.h"
#include "driverlib/gpio.h"
#include "drivers/CTS_structure.h"

//*****************************************************************************
//
// Element structure for the "down" button.
//
//*****************************************************************************
const tCapTouchElement g_sVolumeDownElement =
{
    GPIO_PORTA_AHB_BASE,
    GPIO_PIN_7,
    100,
    2000
};

//*****************************************************************************
//
// Element structure for the "left" button.
//
//*****************************************************************************
const tCapTouchElement g_sLeftElement =
{
    GPIO_PORTA_AHB_BASE,
    GPIO_PIN_6,
    100,
    2000
};

//*****************************************************************************
//
// Element structure for the "right" button.
//
//*****************************************************************************
const tCapTouchElement g_sRightElement =
{
    GPIO_PORTA_AHB_BASE,
    GPIO_PIN_2,
    100,
    2000
};

//*****************************************************************************
//
// Element structure for the "up" button.
//
//*****************************************************************************
const tCapTouchElement g_sVolumeUpElement =
{
    GPIO_PORTA_AHB_BASE,
    GPIO_PIN_3,
    100,
    2000
};

//*****************************************************************************
//
// Element structure for the "middle" button.
//
//*****************************************************************************
const tCapTouchElement g_sMiddleElement =
{
    GPIO_PORTA_AHB_BASE,
    GPIO_PIN_4,
    100,
    1800
};

//*****************************************************************************
//
// Sensor structure for the wheel.
//
//*****************************************************************************
const tSensor g_sSensorWheel =
{
    4,
    100,
    64,
    75,
    0,
    {
        &g_sVolumeUpElement,
        &g_sRightElement,
        &g_sVolumeDownElement,
        &g_sLeftElement
    }
};

//*****************************************************************************
//
// Sensor structure for the middle button.
//
//*****************************************************************************
const tSensor g_sMiddleButton =
{
    1,
    100,
    0,
    0,
    4,
    {
        &g_sMiddleElement
    }
};

