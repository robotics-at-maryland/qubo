//*****************************************************************************
//
// calibrate.c - Calibration routine for the touch screen driver.
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
// This is part of revision 2.1.3.156 of the DK-TM4C129X Firmware Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "grlib/grlib.h"
#include "grlib/widget.h"
#include "utils/ustdlib.h"
#include "drivers/frame.h"
#include "drivers/kentec320x240x16_ssd2119.h"
#include "drivers/pinout.h"
#include "drivers/touch.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Calibration for the Touch Screen (calibrate)</h1>
//!
//! The touch screen driver (../drivers/touch.c) has default parameters,
//! represented by the array g_pi32TouchParameters, to align the touch screen's
//! cooordinates to the display coordinates.  But these parameters might need
//! calibration to achieve a better alignment, of the touch and display
//! coordiantes.
//!
//! The raw sample interface of the touch screen driver is used to compute the
//! calibration matrix required to convert raw samples into screen X/Y
//! positions.  The produced calibration matrix can be inserted into the touch
//! screen driver to map the raw samples into screen coordinates.
//!
//! The touch screen calibration is performed according to the algorithm
//! described by Carlos E. Videles in the June 2002 issue of Embedded Systems
//! Design.  It can be found online at
//! <a href="http://www.embedded.com/story/OEG20020529S0046">
//! http://www.embedded.com/story/OEG20020529S0046</a>.
//
//*****************************************************************************

//*****************************************************************************
//
// Global variables set by the TouchEventHandler and used by the main loop.
//
//*****************************************************************************
volatile bool g_bTouch;
volatile int32_t g_i32CalibX, g_i32CalibY;

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
}
#endif

//*****************************************************************************
//
// The handler function called by the touch driver when touch screen events
// occur.
//
//*****************************************************************************
int32_t TouchEventHandler(uint32_t ui32Message, int32_t i32X, int32_t i32Y)
{
    //
    // Check if pen is up, i.e. touch screen is touched and the pen is lifted.
    //
    if(ui32Message == WIDGET_MSG_PTR_UP)
    {
        //
        // Store the X and Y coordinate values returned by the touch driver to
        // pass to the main loop.
        //
        g_i32CalibX = i32X;
        g_i32CalibY = i32Y;

        //
        // Let the main loop know that the screen has been touched and new
        // coordinate values are received.
        //
        g_bTouch = 1;
    }

    //
    // The return value is not used by the touch driver.  For now returning 1.
    //
    return 1;
}

//*****************************************************************************
//
// Performs calibration of the touch screen.
//
//*****************************************************************************
int
main(void)
{
    int32_t i32Idx, ppi32Points[3][4];
    uint32_t ui32SysClock;
    char pcBuffer[32];
    tContext sContext;
    tRectangle sRect;

    //
    // Run from the PLL at 120 MHz.
    //
    ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                           SYSCTL_OSC_MAIN | SYSCTL_USE_PLL |
                                           SYSCTL_CFG_VCO_480), 120000000);

    //
    // Configure the device pins.
    //
    PinoutSet();

    //
    // Initialize the display driver.
    //
    Kentec320x240x16_SSD2119Init(ui32SysClock);

    //
    // Initialize the graphics context.
    //
    GrContextInit(&sContext, &g_sKentec320x240x16_SSD2119);

    //
    // Draw the application frame.
    //
    FrameDraw(&sContext, "calibrate");

    //
    // Print the instructions across the middle of the screen in white with a
    // 20 point small-caps font.
    //
    GrContextForegroundSet(&sContext, ClrWhite);
    GrContextFontSet(&sContext, g_psFontCmsc20);
    GrStringDrawCentered(&sContext, "Touch the box", -1,
                         GrContextDpyWidthGet(&sContext) / 2,
                         (GrContextDpyHeightGet(&sContext) / 2) - 10, 0);

    //
    // Set the points used for calibration based on the size of the screen.
    //
    ppi32Points[0][0] = GrContextDpyWidthGet(&sContext) / 10;
    ppi32Points[0][1] = (GrContextDpyHeightGet(&sContext) * 2) / 10;
    ppi32Points[1][0] = GrContextDpyWidthGet(&sContext) / 2;
    ppi32Points[1][1] = (GrContextDpyHeightGet(&sContext) * 9) / 10;
    ppi32Points[2][0] = (GrContextDpyWidthGet(&sContext) * 9) / 10;
    ppi32Points[2][1] = GrContextDpyHeightGet(&sContext) / 2;

    //
    // Initialize the touch screen driver and register a callback function to
    // respond to touch screen events.
    //
    TouchScreenInit(ui32SysClock);
    TouchScreenCallbackSet(TouchEventHandler);

    //
    // Loop through the calibration points.
    //
    for(i32Idx = 0; i32Idx < 3; i32Idx++)
    {
        //
        // Fill a white box around the calibration point.
        //
        GrContextForegroundSet(&sContext, ClrWhite);
        sRect.i16XMin = ppi32Points[i32Idx][0] - 5;
        sRect.i16YMin = ppi32Points[i32Idx][1] - 5;
        sRect.i16XMax = ppi32Points[i32Idx][0] + 5;
        sRect.i16YMax = ppi32Points[i32Idx][1] + 5;
        GrRectFill(&sContext, &sRect);

        //
        // Flush any cached drawing operations.
        //
        GrFlush(&sContext);

        //
        // Reset the flag that is set when the screen is touched.
        //
        g_bTouch = 0;

        //
        // Wait until the screen is touched and the pen lifted.
        //
        while(g_bTouch == 0)
        {
        }

        //
        // Store the X and Y coordinates returned by the touch driver.
        //
        ppi32Points[i32Idx][2] = g_i32CalibX;
        ppi32Points[i32Idx][3] = g_i32CalibY;

        //
        // Erase the box around this calibration point.
        //
        GrContextForegroundSet(&sContext, ClrBlack);
        GrRectFill(&sContext, &sRect);
    }

    //
    // Clear the screen.
    //
    sRect.i16XMin = 0;
    sRect.i16YMin = 0;
    sRect.i16XMax = GrContextDpyWidthGet(&sContext) - 1;
    sRect.i16YMax = GrContextDpyHeightGet(&sContext) - 1;
    GrRectFill(&sContext, &sRect);

    //
    // Indicate that the calibration data is being displayed.
    //
    GrContextForegroundSet(&sContext, ClrWhite);
    GrStringDraw(&sContext, "Calibration data:", -1, 16, 32, 0);

    //
    // Compute and display the M0 calibration value.
    //
    usprintf(pcBuffer, "M0 = %d",
             (((ppi32Points[0][0] - ppi32Points[2][0]) *
               (ppi32Points[1][3] - ppi32Points[2][3])) -
              ((ppi32Points[1][0] - ppi32Points[2][0]) *
               (ppi32Points[0][3] - ppi32Points[2][3]))));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 72, 0);

    //
    // Compute and display the M1 calibration value.
    //
    usprintf(pcBuffer, "M1 = %d",
             (((ppi32Points[0][2] - ppi32Points[2][2]) *
               (ppi32Points[1][0] - ppi32Points[2][0])) -
              ((ppi32Points[0][0] - ppi32Points[2][0]) *
               (ppi32Points[1][2] - ppi32Points[2][2]))));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 92, 0);

    //
    // Compute and display the M2 calibration value.
    //
    usprintf(pcBuffer, "M2 = %d",
             ((((ppi32Points[2][2] * ppi32Points[1][0]) -
                (ppi32Points[1][2] * ppi32Points[2][0])) * ppi32Points[0][3]) +
              (((ppi32Points[0][2] * ppi32Points[2][0]) -
                (ppi32Points[2][2] * ppi32Points[0][0])) * ppi32Points[1][3]) +
              (((ppi32Points[1][2] * ppi32Points[0][0]) -
                (ppi32Points[0][2] * ppi32Points[1][0])) *
               ppi32Points[2][3])));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 112, 0);

    //
    // Compute and display the M3 calibration value.
    //
    usprintf(pcBuffer, "M3 = %d",
             (((ppi32Points[0][1] - ppi32Points[2][1]) *
               (ppi32Points[1][3] - ppi32Points[2][3])) -
              ((ppi32Points[1][1] - ppi32Points[2][1]) *
               (ppi32Points[0][3] - ppi32Points[2][3]))));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 132, 0);

    //
    // Compute and display the M4 calibration value.
    //
    usprintf(pcBuffer, "M4 = %d",
             (((ppi32Points[0][2] - ppi32Points[2][2]) *
               (ppi32Points[1][1] - ppi32Points[2][1])) -
              ((ppi32Points[0][1] - ppi32Points[2][1]) *
               (ppi32Points[1][2] - ppi32Points[2][2]))));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 152, 0);

    //
    // Compute and display the M5 calibration value.
    //
    usprintf(pcBuffer, "M5 = %d",
             ((((ppi32Points[2][2] * ppi32Points[1][1]) -
                (ppi32Points[1][2] * ppi32Points[2][1])) * ppi32Points[0][3]) +
              (((ppi32Points[0][2] * ppi32Points[2][1]) -
                (ppi32Points[2][2] * ppi32Points[0][1])) * ppi32Points[1][3]) +
              (((ppi32Points[1][2] * ppi32Points[0][1]) -
                (ppi32Points[0][2] * ppi32Points[1][1])) *
               ppi32Points[2][3])));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 172, 0);

    //
    // Compute and display the M6 calibration value.
    //
    usprintf(pcBuffer, "M6 = %d",
             (((ppi32Points[0][2] - ppi32Points[2][2]) *
               (ppi32Points[1][3] - ppi32Points[2][3])) -
              ((ppi32Points[1][2] - ppi32Points[2][2]) *
               (ppi32Points[0][3] - ppi32Points[2][3]))));
    GrStringDraw(&sContext, pcBuffer, -1, 16, 192, 0);

    //
    // Flush any cached drawing operations.
    //
    GrFlush(&sContext);

    //
    // The calibration is complete.  Sit around and wait for a reset.
    //
    while(1)
    {
    }
}
