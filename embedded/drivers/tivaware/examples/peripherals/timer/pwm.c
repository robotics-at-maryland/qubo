//*****************************************************************************
//
// pwm.c - Example demonstrating timer-based PWM on a 16-bit CCP.
//
// Copyright (c) 2010-2016 Texas Instruments Incorporated.  All rights reserved.
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

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_gpio.h"
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_timer.h"
#include "inc/hw_types.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"

//*****************************************************************************
//
//! \addtogroup timer_examples_list
//! <h1>PWM using Timer (pwm)</h1>
//!
//! This example shows how to configure Timer1B to generate a PWM signal on the
//! timer's CCP pin.
//!
//! This example uses the following peripherals and I/O signals.  You must
//! review these and change as needed for your own board:
//! - TIMER1 peripheral
//! - GPIO Port B peripheral (for T1CCP1 pin)
//! - T1CCP1 - PB5
//!
//! The following UART signals are configured only for displaying console
//! messages for this example.  These are not required for operation of
//! Timer0.
//! - UART0 peripheral
//! - GPIO Port A peripheral (for UART0 pins)
//! - UART0RX - PA0
//! - UART0TX - PA1
//!
//! This example uses the following interrupt handlers.  To use this example
//! in your own application you must add these interrupt handlers to your
//! vector table.
//! - None.
//
//*****************************************************************************

//*****************************************************************************
//
// The g_ui32SysClock contains the system clock frequency
//
//*****************************************************************************
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    uint32_t g_ui32SysClock;
#endif

//*****************************************************************************
//
// This function sets up UART0 to be used for a console to display information
// as the example is running.
//
//*****************************************************************************
void
InitConsole(void)
{
    //
    // Enable GPIO port A which is used for UART0 pins.
    // TODO: change this to whichever GPIO port you are using.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    //
    // Configure the pin muxing for UART0 functions on port A0 and A1.
    // This step is not necessary if your part does not support pin muxing.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);

    //
    // Enable UART0 so that we can configure the clock.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Use the internal 16MHz oscillator as the UART clock source.
    //
    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    //
    // Select the alternate (UART) function for these pins.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, 16000000);
}

//*****************************************************************************
//
// Prints out 5x "." with a second delay after each print.  This function will
// then backspace, clear the previously printed dots, backspace again so you
// continuously printout on the same line.
//
//*****************************************************************************
void
PrintRunningDots(void)
{
    UARTprintf(". ");
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    SysCtlDelay(g_ui32SysClock / 3);
#else
    SysCtlDelay(SysCtlClockGet() / 3);
#endif
    UARTprintf(". ");
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    SysCtlDelay(g_ui32SysClock / 3);
#else
    SysCtlDelay(SysCtlClockGet() / 3);
#endif
    UARTprintf(". ");
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    SysCtlDelay(g_ui32SysClock / 3);
#else
    SysCtlDelay(SysCtlClockGet() / 3);
#endif
    UARTprintf(". ");
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    SysCtlDelay(g_ui32SysClock / 3);
#else
    SysCtlDelay(SysCtlClockGet() / 3);
#endif
    UARTprintf(". ");
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    SysCtlDelay(g_ui32SysClock / 3);
#else
    SysCtlDelay(SysCtlClockGet() / 3);
#endif
    UARTprintf("\b\b\b\b\b\b\b\b\b\b");
    UARTprintf("          ");
    UARTprintf("\b\b\b\b\b\b\b\b\b\b");
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    SysCtlDelay(g_ui32SysClock / 3);
#else
    SysCtlDelay(SysCtlClockGet() / 3);
#endif
}

//*****************************************************************************
//
// Configure Timer1B as a 16-bit PWM with a duty cycle of 66%.
//
//*****************************************************************************
int
main(void)
{
    //
    // Set the clocking to run directly from the external crystal/oscillator.
    // TODO: The SYSCTL_XTAL_ value must be changed to match the value of the
    // crystal on your board.
    //
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    g_ui32SysClock = SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                       SYSCTL_OSC_MAIN |
                                       SYSCTL_USE_OSC), 25000000);
#else
    SysCtlClockSet(SYSCTL_SYSDIV_1 | SYSCTL_USE_OSC | SYSCTL_OSC_MAIN |
                   SYSCTL_XTAL_16MHZ);
#endif

    //
    // The Timer1 peripheral must be enabled for use.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER1);

    //
    // For this example T1CCP1 is used with port B pin 5.
    // The actual port and pins used may be different on your part, consult
    // the data sheet for more information.
    // GPIO port B needs to be enabled so these pins can be used.
    // TODO: change this to whichever GPIO port you are using.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);

    //
    // Configure the GPIO pin muxing for the Timer/CCP function.
    // This is only necessary if your part supports GPIO pin function muxing.
    // Study the data sheet to see which functions are allocated per pin.
    // TODO: change this to select the port/pin you are using
    //
    GPIOPinConfigure(GPIO_PB5_T1CCP1);

    //
    // Set up the serial console to use for displaying messages.  This is
    // just for this example program and is not needed for Timer/PWM operation.
    //
    InitConsole();

    //
    // Configure the ccp settings for CCP pin.  This function also gives
    // control of these pins to the SSI hardware.  Consult the data sheet to
    // see which functions are allocated per pin.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinTypeTimer(GPIO_PORTB_BASE, GPIO_PIN_5);

    //
    // Display the example setup on the console.
    //
    UARTprintf("16-Bit Timer PWM ->");
    UARTprintf("\n   Timer = Timer1B");
    UARTprintf("\n   Mode = PWM");
    UARTprintf("\n   Duty Cycle = 66%%\n");
    UARTprintf("\nGenerating PWM on CCP3 (PE4) -> ");

    //
    // Configure Timer1B as a 16-bit periodic timer.
    //
    TimerConfigure(TIMER1_BASE, TIMER_CFG_SPLIT_PAIR | TIMER_CFG_B_PWM);

    //
    // Set the Timer1B load value to 50000.  For this example a 66% duty cycle
    // PWM signal will be generated.  From the load value (i.e. 50000) down to
    // match value (set below) the signal will be high.  From the match value
    // to 0 the timer will be low.
    //
    TimerLoadSet(TIMER1_BASE, TIMER_B, 50000);

    //
    // Set the Timer1B match value to load value / 3.
    //
    TimerMatchSet(TIMER1_BASE, TIMER_B,
                  TimerLoadGet(TIMER1_BASE, TIMER_B) / 3);

    //
    // Enable Timer1B.
    //
    TimerEnable(TIMER1_BASE, TIMER_B);

    //
    // Loop forever while the Timer1B PWM runs.
    //
    while(1)
    {
        //
        // Print out indication on the console that the program is running.
        //
        PrintRunningDots();
    }
}
