//*****************************************************************************
//
// io.c - I/O routines for the cc3100_http_server example application.
//
// Copyright (c) 2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
//
//*****************************************************************************
#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/gpio.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/interrupt.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "utils/ustdlib.h"
#include "io.h"

//*****************************************************************************
//
// Hardware connection for the Toggle LED.
//
//*****************************************************************************
#define LED_PORT_BASE GPIO_PORTN_BASE
#define LED_PIN GPIO_PIN_0

//*****************************************************************************
//
// Hardware connection for the Animation LED.
//
//*****************************************************************************
#define LED_ANIM_PORT_BASE GPIO_PORTN_BASE
#define LED_ANIM_PIN GPIO_PIN_1

//*****************************************************************************
//
// The current speed of the animation LED expressed as a percentage.  By
// default it is set to 10%.
//
//*****************************************************************************
volatile uint32_t g_ui32AnimSpeed = 10;

//*****************************************************************************
//
// The system clock speed.
//
//*****************************************************************************
extern uint32_t g_SysClock;

//*****************************************************************************
//
// The interrupt handler for the timer used to pace the animation LED.
//
//*****************************************************************************
void
AnimTimerIntHandler(void)
{
    //
    // Clear the timer interrupt.
    //
    MAP_TimerIntClear(TIMER2_BASE, TIMER_TIMA_TIMEOUT);

    //
	// Toggle the GPIO
	//
	MAP_GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_1,
					 (MAP_GPIOPinRead(GPIO_PORTN_BASE, GPIO_PIN_1) ^
					  GPIO_PIN_1));
}

//*****************************************************************************
//
// Set the timer used to pace the animation.  We scale the timer timeout such
// that a speed of 100% causes the timer to tick once every 20 mS (50Hz).
//
//*****************************************************************************
void
io_set_timer(uint32_t ui32SpeedPercent)
{
    uint32_t ui32Timeout;

    //
    // Turn the timer off while we are configuring it.
    //
    MAP_TimerDisable(TIMER2_BASE, TIMER_A);

    //
    // If the speed is non-zero, we reset the timeout.  If it is zero, we
    // just leave the timer disabled.
    //
    if(ui32SpeedPercent)
    {
        //
        // Set Timeout (or timer compare value)
        //
        ui32Timeout = g_SysClock / 50;
        ui32Timeout = (ui32Timeout * 100 ) / ui32SpeedPercent;

        MAP_TimerLoadSet(TIMER2_BASE, TIMER_A, ui32Timeout);
        MAP_TimerEnable(TIMER2_BASE, TIMER_A);
    }
}

//*****************************************************************************
//
// Initialize the IO used in this demo
//
//*****************************************************************************
void
io_init(void)
{
    //
    // Configure Port N0 for as an output for the status LED.
    //
    MAP_GPIOPinTypeGPIOOutput(LED_PORT_BASE, LED_PIN);

    //
    // Configure Port N0 for as an output for the animation LED.
    //
    MAP_GPIOPinTypeGPIOOutput(LED_ANIM_PORT_BASE, LED_ANIM_PIN);

    //
    // Initialize LED to OFF (0)
    //
    MAP_GPIOPinWrite(LED_PORT_BASE, LED_PIN, 0);

    //
    // Initialize animation LED to OFF (0)
    //
    MAP_GPIOPinWrite(LED_ANIM_PORT_BASE, LED_ANIM_PIN, 0);

    //
    // Enable the peripherals used by this example.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER2);

    //
    // Configure the timer used to pace the animation.
    //
    MAP_TimerConfigure(TIMER2_BASE, TIMER_CFG_PERIODIC);

    //
    // Setup the interrupts for the timer timeouts.
    //
    MAP_IntEnable(INT_TIMER2A);
    MAP_TimerIntEnable(TIMER2_BASE, TIMER_TIMA_TIMEOUT);

    //
    // Set the timer for the current animation speed.  This enables the
    // timer as a side effect.
    //
    io_set_timer(g_ui32AnimSpeed);
}

//*****************************************************************************
//
// Set the status LED on or off.
//
//*****************************************************************************
void
io_set_led(bool bOn)
{
    //
    // Turn the LED on or off as requested.
    //
    MAP_GPIOPinWrite(LED_PORT_BASE, LED_PIN, bOn ? LED_PIN : 0);
}

//*****************************************************************************
//
// Set the speed of the animation shown on the display.
//
//*****************************************************************************
void
io_set_animation_speed(uint32_t ui32Speed)
{
    //
    // If the number is valid, set the new speed.
    //
    if(ui32Speed <= 100)
    {
        g_ui32AnimSpeed = ui32Speed;
        io_set_timer(g_ui32AnimSpeed);
    }
}
