//*****************************************************************************
//
// edge_count.c - Timer edge count mode example.
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
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/debug.h"
#include "driverlib/fpu.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "grlib/grlib.h"
#include "drivers/cfal96x64x16.h"
#include "utils/ustdlib.h"

//*****************************************************************************
//
//! \addtogroup timer_examples_list
//! <h1>Timer Edge Count (edge_count)</h1>
//!
//! This example application demonstrates the use of a general purpose timer
//! in down edge count mode.  Timer 4 is configured to decrement each time
//! a rising edge is seen on PM0/CCP0.  The count is initialized to 9 and the
//! match is set to 0, causing an interrupt to fire after 10 positive edges
//! detected on the CCP pin.
//
//*****************************************************************************

//*****************************************************************************
//
// Flags that contain the current value of the interrupt indicator as displayed
// on the CSTN display.
//
//*****************************************************************************
uint32_t g_ui32Flags;

//*****************************************************************************
//
// An interrupt counter indicating how often the timer down counter reached
// 0.
//
//*****************************************************************************
volatile uint32_t g_ui32IntCount;

//*****************************************************************************
//
// The graphics context used to draw on the display.
//
//*****************************************************************************
tContext g_sContext;

//*****************************************************************************
//
// A buffer used for string formatting.
//
//*****************************************************************************
#define PRINT_BUFF_SIZE 8
char g_pcPrintBuff[PRINT_BUFF_SIZE];

//*****************************************************************************
//
// In this particular example, this function merely updates an interrupt
// count and sets a flag which tells the main loop to update the display.
//
// TODO: Update or remove this function as required by your specific
// application.
//
//*****************************************************************************
void
ProcessInterrupt(void)
{
    //
    // Update our interrupt counter.
    //
    g_ui32IntCount++;

    //
    // Toggle the interrupt flag telling the main loop to update the display.
    //
    HWREGBITW(&g_ui32Flags, 0) ^= 1;
}

//*****************************************************************************
//
// Initialize the display.  This function is specific to the  EK-LM4F232 board
// and contains nothing directly relevant to the timer configuration or
// operation.
//
//*****************************************************************************
void
InitDisplay(void)
{
    tRectangle sRect;

    //
    // Initialize the display driver.
    //
    CFAL96x64x16Init();

    //
    // Initialize the graphics context and find the middle X coordinate.
    //
    GrContextInit(&g_sContext, &g_sCFAL96x64x16);

    //
    // Fill the top part of the screen with blue to create the banner.
    //
    sRect.i16XMin = 0;
    sRect.i16YMin = 0;
    sRect.i16XMax = GrContextDpyWidthGet(&g_sContext) - 1;
    sRect.i16YMax = 9;
    GrContextForegroundSet(&g_sContext, ClrDarkBlue);
    GrRectFill(&g_sContext, &sRect);

    //
    // Change foreground for white text.
    //
    GrContextForegroundSet(&g_sContext, ClrWhite);

    //
    // Put the application name in the middle of the banner.
    //
    GrContextFontSet(&g_sContext, g_psFontFixed6x8);
    GrStringDrawCentered(&g_sContext, "edge-count", -1,
                         GrContextDpyWidthGet(&g_sContext) / 2, 4, 0);

    //
    // Initialize timer status display.
    //
    GrContextFontSet(&g_sContext, g_psFontFixed6x8);
    GrStringDraw(&g_sContext, "Countdown:", -1, 8, 26, 0);
    GrStringDraw(&g_sContext, "Interrupts:", -1, 8, 36, 0);
}

//*****************************************************************************
//
// The main loop of the application.  This implementation is specific to the
// EK-LM4F232 board and merely displays the current timer count value and the
// number of interrupts taken.  It contains nothing directly relevant to the
// timer configuration or operation.
//
//*****************************************************************************
void
MainLoopRun(void)
{
    uint32_t ui32Count, ui32LastCount;

    //
    // Set up for the main loop.
    //
    ui32LastCount = 10;

    //
    // Loop forever while the timer runs.
    //
    while(1)
    {
        //
        // Get the current timer count.
        //
        ui32Count = ROM_TimerValueGet(TIMER4_BASE, TIMER_A);

        //
        // Has it changed?
        //
        if(ui32Count != ui32LastCount)
        {
            //
            // Yes - update the display.
            //
            usnprintf(g_pcPrintBuff, PRINT_BUFF_SIZE, "%d ", ui32Count);
            GrStringDraw(&g_sContext, g_pcPrintBuff, -1, 80, 26, true);

            //
            // Remember the new count value.
            //
            ui32LastCount = ui32Count;
        }

        //
        // Has there been an interrupt since last we checked?
        //
        if(HWREGBITW(&g_ui32Flags, 0))
        {
            //
            // Clear the bit.
            //
            HWREGBITW(&g_ui32Flags, 0) = 0;

            //
            // Update the interrupt count.
            //
            usnprintf(g_pcPrintBuff, PRINT_BUFF_SIZE, "%d ", g_ui32IntCount);
            GrStringDraw(&g_sContext, g_pcPrintBuff, -1, 80, 36, true);
        }
    }
}

//*****************************************************************************
//
// The interrupt handler for timer 4.  This will be called whenever the timer
// count reaches the match value (0 in this example).
//
// TODO: Make sure you hook your ISR to the correct vector in the application
// startup file.
//
//*****************************************************************************
void
Timer4IntHandler(void)
{
    //
    // Clear the timer interrupt.
    //
    // TODO: Rework this for the timer you are using in your application.
    //
    ROM_TimerIntClear(TIMER4_BASE, TIMER_CAPA_MATCH);

    //
    // TODO: Do whatever your application needs to do when the relevant
    // number of edges have been counted.
    //
    ProcessInterrupt();

    //
    // The timer is automatically stopped when it reaches the match value
    // so re-enable it here.
    //
    // TODO: Whether you reenable the timer here or elsewhere will be up to
    // your particular application.
    //
    ROM_TimerEnable(TIMER4_BASE, TIMER_A);
}

//*****************************************************************************
//
// This example application demonstrates the use of the timers to generate
// periodic interrupts.
//
//*****************************************************************************
int
main(void)
{
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    uint32_t ui32SysClock;
#endif

    //
    // Enable lazy stacking for interrupt handlers.  This allows floating-point
    // instructions to be used within interrupt handlers, but at the expense of
    // extra stack usage.
    //
    ROM_FPULazyStackingEnable();

    //
    // Set the clocking to run directly from the crystal.
    // TODO: Set the system clock appropriately for your application.
    //
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    ui32SysClock = SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                       SYSCTL_OSC_MAIN |
                                       SYSCTL_USE_OSC), 25000000);
#else
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_1 | SYSCTL_USE_OSC | SYSCTL_OSC_MAIN |
                       SYSCTL_XTAL_16MHZ);
#endif

    //
    // Initialize the display on the board.  This is not directly related
    // to the timer operation but makes it easier to see what's going on
    // when you run this on an EK-LM4F232 board.
    //
    // TODO: Remove or replace this call with something appropriate for
    // your hardware.
    //
    InitDisplay();

    //
    // Enable the peripherals used by this example.
    //
    // TODO: Update this depending upon the general purpose timer and
    // CCP pin you intend using.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER4);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOM);

    //
    // Configure PM0 as the CCP0 pin for timer 4.
    //
    // TODO: This is set up to use GPIO PM0 which can be configured
    // as the CCP0 pin for Timer4 and also happens to be attached to
    // a switch on the EK-LM4F232 board.  Change this configuration to
    // correspond to the correct pin for your application.
    //
    ROM_GPIOPinTypeTimer(GPIO_PORTM_BASE, GPIO_PIN_0);
    GPIOPinConfigure(GPIO_PM0_T4CCP0);

    //
    // Set the pin to use the internal pull-up.
    //
    // TODO: Remove or replace this call to correspond to the wiring
    // of the CCP pin you are using.  If your board has an external
    // pull-up or pull-down, this will not be required.
    //
    MAP_GPIOPadConfigSet(GPIO_PORTM_BASE, GPIO_PIN_0,
                     GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

    //
    // Enable processor interrupts.
    //
    ROM_IntMasterEnable();

    //
    // Configure the timers in downward edge count mode.
    //
    // TODO: Modify this to configure the specific general purpose
    // timer you are using.  The timer choice is intimately tied to
    // the pin whose edges you want to capture because specific CCP
    // pins are connected to specific timers.
    //
    ROM_TimerConfigure(TIMER4_BASE, (TIMER_CFG_SPLIT_PAIR |
                       TIMER_CFG_A_CAP_COUNT));
    ROM_TimerControlEvent(TIMER4_BASE, TIMER_A, TIMER_EVENT_POS_EDGE);
    ROM_TimerLoadSet(TIMER4_BASE, TIMER_A, 9);
    ROM_TimerMatchSet(TIMER4_BASE, TIMER_A, 0);

    //
    // Setup the interrupt for the edge capture timer.  Note that we
    // use the capture match interrupt and NOT the timeout interrupt!
    //
    // TODO: Modify to enable the specific timer you are using.
    //
    ROM_IntEnable(INT_TIMER4A);
    ROM_TimerIntEnable(TIMER4_BASE, TIMER_CAPA_MATCH);

    //
    // Enable the timer.
    //
    // TODO: Modify to enable the specific timer you are using.
    //
    ROM_TimerEnable(TIMER4_BASE, TIMER_A);

    //
    // At this point, the timer will count down every time a positive
    // edge is detected on the relevant pin.   When the count reaches
    // 0, the timer count reloads, the interrupt fires and the timer
    // is disabled.  The ISR can then restart the timer if required.
    //
    MainLoopRun();
}
