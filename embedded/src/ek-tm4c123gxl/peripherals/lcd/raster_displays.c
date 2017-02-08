//*****************************************************************************
//
// raster_displays.c - Timing settings for various raster displays
//
// Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
// 
//   Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
//   Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the  
//   distribution.
// 
//   Neither the name of Texas Instruments Incorporated nor the names of
//   its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/lcd.h"
#include "driverlib/ssi.h"
#include "driverlib/rom_map.h"
#include "raster_displays.h"

//*****************************************************************************
//
// Initialization function for the LXD display. This turns on the backlight
// and sets the correct RGB mode.
//
//*****************************************************************************
void
InitLXDAndFormike(uint32_t ui32SysClk)
{
    //
    // Convert PT2 & 3 back to a GPIO output (they were LCDDATA18/19)
    //
    HWREG(GPIO_PORTT_BASE + GPIO_O_PCTL) &= 0xFFFF00FF;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTT_BASE, GPIO_PIN_2 | GPIO_PIN_3);

    //
    // Convert PS0 and 1 back to GPIO outputs (they were LCDDATA20/21)
    //
    HWREG(GPIO_PORTS_BASE + GPIO_O_PCTL) &= 0xFFFFFF00;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTS_BASE, GPIO_PIN_1 | GPIO_PIN_0);

    //
    // Drive PS0/1 to set the display TL to BR scanning direction.
    //
    MAP_GPIOPinWrite(GPIO_PORTS_BASE, GPIO_PIN_1 | GPIO_PIN_0, GPIO_PIN_0);

    //
    // Write PT2 high to turn on the backlight.
    //
    MAP_GPIOPinWrite(GPIO_PORTT_BASE, GPIO_PIN_2, GPIO_PIN_2);

    //
    // Turn OE/DE off
    //
    MAP_GPIOPinTypeGPIOInput(GPIO_PORTJ_BASE, GPIO_PIN_6);

    //
    // Write PT3 low to set the display into RGB mode.
    //
    MAP_GPIOPinWrite(GPIO_PORTT_BASE, GPIO_PIN_3, 0);

    //
    // Write PT3 low to set the display into RGB mode.
    //
    MAP_GPIOPinWrite(GPIO_PORTT_BASE, GPIO_PIN_3, 0);
}

//*****************************************************************************
//
// Initialization function for the InnoLux display. This turns on the backlight
// after the required 200mS delay.
//
// VLED control - PD6
// LED_EN       - PE5
// LED_PWM      - PR2 or PE4
//
//*****************************************************************************
void
InitInnoLux(uint32_t ui32SysClk)
{
    //
    // Turn on GPIO Ports D, E and R.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);
#ifdef USE_PR2_FOR_PWM
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOR);
#endif

    //
    // Configure PD6 (VLED) as an output.
    //
    HWREG(GPIO_PORTD_BASE + GPIO_O_PCTL) &= 0xF0FFFFFF;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_6);

    //
    // Configure PE5 and PR2 as outputs.
    //
    HWREG(GPIO_PORTE_BASE + GPIO_O_PCTL) &= 0xFF0FFFFF;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_5);
#ifdef USE_PR2_FOR_PWM
    HWREG(GPIO_PORTR_BASE + GPIO_O_PCTL) &= 0xFFFFF0FF;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTR_BASE, GPIO_PIN_2);
    GPIOPadConfigSet(GPIO_PORTR_BASE, GPIO_PIN_2, GPIO_STRENGTH_12MA, GPIO_PIN_TYPE_STD);
#else
    HWREG(GPIO_PORTE_BASE + GPIO_O_PCTL) &= 0xFFF0FFFF;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_4);
#endif

    //
    // Drive the enables low for now.
    //
    MAP_GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_6, 0);
#ifdef USE_PR2_FOR_PWM
    MAP_GPIOPinWrite(GPIO_PORTR_BASE, GPIO_PIN_2, 0);
    MAP_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, 0);
#else
    MAP_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5 | GPIO_PIN_4, 0);
#endif


    //
    // Wait 200mS.  This delay is specified in the display datasheet.
    //
    MAP_SysCtlDelay((ui32SysClk / 5) / 3);

    //
    // Drive PD6 high to enable VLED to the display.
    //
    MAP_GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_6, GPIO_PIN_6);

    //
    // Wait another 10mS before we turn on the PWM to set the display
    // backlight brightness.
    //
    MAP_SysCtlDelay((ui32SysClk / 100) / 3);

    //
    // Start the LED backlight signaling by asserting LED-PWM (PR2).
    //
#ifdef USE_PR2_FOR_PWM
    MAP_GPIOPinWrite(GPIO_PORTR_BASE, GPIO_PIN_2, GPIO_PIN_2);
#else
    MAP_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_4, GPIO_PIN_4);
#endif

    //
    // Wait another 10mS before returning to ensure that the raster is not
    // enabled too soon.
    //
    MAP_SysCtlDelay((ui32SysClk / 100) / 3);

    //
    // Enable the LED backlight by asserting LED-EN (PE5).
    //
    MAP_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, GPIO_PIN_5);
}

//*****************************************************************************
//
// Initialization function for the Formike 800x480 display. This turns on the
// backlight.
//
//*****************************************************************************
void
EnableBacklightOnPT2(uint32_t ui32SysClk)
{
    //
    // Convert PT2 back to a GPIO output (it was LCDDATA18)
    //
    HWREG(GPIO_PORTT_BASE + GPIO_O_PCTL) &= 0xFFFFF0FF;
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTT_BASE, GPIO_PIN_2);

    //
    // Write the pin high to turn on the backlight.
    //
    MAP_GPIOPinWrite(GPIO_PORTT_BASE, GPIO_PIN_2, GPIO_PIN_2);
}


//*****************************************************************************
//
// Video interface timings for an Optrex T-55226D043J-LW-A-AAN display with
// 800x480 resolution refreshed at 75Hz.
//
//*****************************************************************************
const tRasterDisplayInfo g_sOptrex800x480x75Hz =
{
    "800x480 at 75Hz on Optrex T-55226D043J-LW-A-AAN",
    30000000,
    SYSCTL_CFG_VCO_480,
    120000000,
    {
        (RASTER_TIMING_ACTIVE_HIGH_PIXCLK |
         RASTER_TIMING_SYNCS_ON_RISING_PIXCLK),
        800, 480,
        2, 30, 8,
        10, 10, 8,
        0
    },
    0
};

//*****************************************************************************
//
// Video interface timings for a Formike KWH070KQ13 display with 800x480
// resolution, refreshed at 60Hz.
//
//*****************************************************************************
const tRasterDisplayInfo g_sFormike800x480x60Hz =
{
    "800x480 at 60Hz on Formike KWH070KQ13",
    40000000,
    SYSCTL_CFG_VCO_480,
    120000000,
    {
        (RASTER_TIMING_ACTIVE_HIGH_PIXCLK |
        RASTER_TIMING_SYNCS_ON_RISING_PIXCLK |
        RASTER_TIMING_ACTIVE_LOW_HSYNC |
        RASTER_TIMING_ACTIVE_LOW_VSYNC |
        RASTER_TIMING_ACTIVE_HIGH_OE),
        800, 480,
        210, 45, 1,
        133, 22, 1,
        0
    },
    InitLXDAndFormike
};

//*****************************************************************************
//
// Video interface timings for an Innolux EJ090NA-03A display with 800x480
// resolution, refreshed at 60Hz.
//
//*****************************************************************************
const tRasterDisplayInfo g_sInnoLux800x480x60Hz =
{
    "800x480 at 60Hz on InnoLux EJ090NA-03A",
    34000000,
    SYSCTL_CFG_VCO_480,
    120000000,
    {
        (RASTER_TIMING_ACTIVE_HIGH_PIXCLK |
        RASTER_TIMING_SYNCS_ON_RISING_PIXCLK |
        RASTER_TIMING_ACTIVE_LOW_HSYNC |
        RASTER_TIMING_ACTIVE_LOW_VSYNC |
        RASTER_TIMING_ACTIVE_HIGH_OE),
        800, 480,
        128, 128, 16,
        22, 23, 2,
        0
    },
    InitInnoLux
};

//*****************************************************************************
//
// Video interface timings for an LXD M7170A display with 640x480 resolution,
// refreshed at 58Hz.
//
//*****************************************************************************
const tRasterDisplayInfo g_sLXD640x480x60Hz =
{
    "640x480 at 58Hz on LXD M7170A",
    24000000,
    SYSCTL_CFG_VCO_480,
    120000000,
    {
        (RASTER_TIMING_ACTIVE_LOW_PIXCLK |
         RASTER_TIMING_SYNCS_ON_FALLING_PIXCLK |
         RASTER_TIMING_ACTIVE_LOW_HSYNC |
         RASTER_TIMING_ACTIVE_LOW_VSYNC |
         RASTER_TIMING_ACTIVE_HIGH_OE),
        640, 480,
        16, 134, 10,
        32, 11, 2,
        0
    },
    InitLXDAndFormike
};

