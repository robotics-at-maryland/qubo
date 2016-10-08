//*****************************************************************************
//
// airmouse.c - Main routines for SensHub Air Mouse Demo.
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
#include <stdio.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/debug.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "drivers/rgb.h"
#include "remoti_uart.h"
#include "remoti_npi.h"
#include "remoti_rti.h"
#include "remoti_rtis.h"
#include "drivers/buttons.h"
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "usblib/device/usbdevice.h"
#include "usblib/device/usbdcomp.h"
#include "usblib/device/usbdhid.h"
#include "usblib/device/usbdhidmouse.h"
#include "usblib/device/usbdhidkeyb.h"
#include "events.h"
#include "motion.h"
#include "usb_structs.h"
#include "lprf.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Motion Air Mouse (airmouse)</h1>
//!
//! This example demonstrates the use of the Sensor Library, TM4C123G LaunchPad
//! and the SensHub BoosterPack to fuse nine axis sensor measurements into
//! motion and gesture events.  These events are then transformed into mouse
//! and keyboard events to perform standard HID tasks.
//!
//! Connect the device USB port on the side of the LaunchPad to a standard
//! computer USB port.  The LaunchPad with SensHub BoosterPack enumerates on
//! the USB bus as a composite HID keyboard and mouse.
//!
//! Hold the LaunchPad with the buttons away from the user and toward the
//! computer with USB Device cable exiting the right and bottom corner of the
//! board.
//!
//! - Roll or tilt the LaunchPad to move the mouse cursor of the computer
//! up, down, left and right.
//!
//! - The buttons on the LaunchPad perform the left and right mouse click
//! actions.  The buttons on the SensHub BoosterPack are not currently used by
//! this example.
//!
//! - A quick spin of the LaunchPad generates a PAGE_UP or PAGE_DOWN
//! keyboard press and release depending on the direction of the spin.  This
//! motion simulates scrolling.
//!
//! - A quick horizontal jerk to the left or right  generates a CTRL+ or
//! CTRL- keyboard event, which creates the zoom effect used in many
//! applications, especially web browsers.
//!
//! - A quick vertical lift generates an ALT+TAB keyboard event, which
//! allows the computer user to select between currently open windows.
//!
//! - A quick twist to the left or right moves the window selector.
//!
//! - A quick jerk in the down direction selects the desired window and
//! closes the window selection dialog.
//!
//! This example also supports the RemoTI low power RF Zigbee&reg;&nbsp;human
//! interface device profile.  The wireless features of this example require the
//! CC2533EMK expansion card and the CC2531EMK USB Dongle.  For details and
//! instructions for wireless operations see the Wiki at
//! http://processors.wiki.ti.com/index.php/Tiva_C_Series_LaunchPad and
//! http://processors.wiki.ti.com/index.php/Wireless_Air_Mouse_Guide.
//
//*****************************************************************************

//*****************************************************************************
//
// Holds command bits used to signal the main loop to perform various tasks.
//
//*****************************************************************************
volatile uint32_t g_pui32RGBColors[3];

//*****************************************************************************
//
// Holds command bits used to signal the main loop to perform various tasks.
//
//*****************************************************************************
volatile uint_fast32_t g_ui32Events;

//*****************************************************************************
//
// Hold the state of the buttons on the board.
//
//*****************************************************************************
volatile uint_fast8_t g_ui8Buttons;

//*****************************************************************************
//
// Global system tick counter holds elapsed time since the application started
// expressed in 100ths of a second.
//
//*****************************************************************************
volatile uint_fast32_t g_ui32SysTickCount;

//*****************************************************************************
//
// The memory allocated to hold the composite descriptor that is created by
// the call to USBDCompositeInit().
//
//*****************************************************************************
#define DESCRIPTOR_DATA_SIZE    (COMPOSITE_DHID_SIZE + COMPOSITE_DHID_SIZE)
uint8_t g_pui8DescriptorData[DESCRIPTOR_DATA_SIZE];


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
// This is the interrupt handler for the SysTick interrupt.  It is called
// periodically and updates a global tick counter then sets a flag to tell the
// main loop to move the mouse.
//
//*****************************************************************************
void
SysTickIntHandler(void)
{
    g_ui32SysTickCount++;
    HWREGBITW(&g_ui32Events, USB_TICK_EVENT) = 1;
    HWREGBITW(&g_ui32Events, LPRF_TICK_EVENT) = 1;
    g_ui8Buttons = ButtonsPoll(0, 0);
}

//*****************************************************************************
//
// Configure the UART and its pins.  This must be called before UARTprintf().
//
//*****************************************************************************
void
ConfigureUART(void)
{
    //
    // Enable the GPIO Peripheral used by the UART.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    //
    // Enable UART0
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Configure GPIO Pins for UART mode.
    //
    ROM_GPIOPinConfigure(GPIO_PA0_U0RX);
    ROM_GPIOPinConfigure(GPIO_PA1_U0TX);
    ROM_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // Use the internal 16MHz oscillator as the UART clock source.
    //
    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, 16000000);
}

//*****************************************************************************
//
// This is the main loop that runs the application.
//
//*****************************************************************************
int
main(void)
{
    //
    // Turn on stacking of FPU registers if FPU is used in the ISR.
    //
    FPULazyStackingEnable();

    //
    // Set the clocking to run from the PLL at 40MHz.
    //
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_5 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN |
                       SYSCTL_XTAL_16MHZ);

    //
    // Set the system tick to fire 100 times per second.
    //
    ROM_SysTickPeriodSet(ROM_SysCtlClockGet() / SYSTICKS_PER_SECOND);
    ROM_SysTickIntEnable();
    ROM_SysTickEnable();

    //
    // Enable the Debug UART.
    //
    ConfigureUART();

    //
    // Print the welcome message to the terminal.
    //
    UARTprintf("\033[2JAir Mouse Application\n");

    //
    // Configure desired interrupt priorities. This makes certain that the DCM
    // is fed data at a consistent rate. Lower numbers equal higher priority.
    //
    ROM_IntPrioritySet(INT_I2C3, 0x00);
    ROM_IntPrioritySet(INT_GPIOB, 0x10);
    ROM_IntPrioritySet(FAULT_SYSTICK, 0x20);
    ROM_IntPrioritySet(INT_UART1, 0x60);
    ROM_IntPrioritySet(INT_UART0, 0x70);
    ROM_IntPrioritySet(INT_WTIMER5B, 0x80);

    //
    // Configure the USB D+ and D- pins.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    ROM_GPIOPinTypeUSBAnalog(GPIO_PORTD_BASE, GPIO_PIN_5 | GPIO_PIN_4);

    //
    // Pass the USB library our device information, initialize the USB
    // controller and connect the device to the bus.
    //
    USBDHIDMouseCompositeInit(0, &g_sMouseDevice, &g_psCompDevices[0]);
    USBDHIDKeyboardCompositeInit(0, &g_sKeyboardDevice, &g_psCompDevices[1]);

    //
    // Set the USB stack mode to Force Device mode.
    //
    USBStackModeSet(0, eUSBModeForceDevice, 0);

    //
    // Pass the device information to the USB library and place the device
    // on the bus.
    //
    USBDCompositeInit(0, &g_sCompDevice, DESCRIPTOR_DATA_SIZE,
                      g_pui8DescriptorData);

    //
    // User Interface Init
    //
    ButtonsInit();
    RGBInit(0);
    RGBEnable();

    //
    // Initialize the motion sub system.
    //
    MotionInit();

    //
    // Initialize the Radio Systems.
    //
    LPRFInit();

    //
    // Drop into the main loop.
    //
    while(1)
    {

        //
        // Check for and handle timer tick events.
        //
        if(HWREGBITW(&g_ui32Events, USB_TICK_EVENT) == 1)
        {
            //
            // Clear the Tick event flag. Set in SysTick interrupt handler.
            //
            HWREGBITW(&g_ui32Events, USB_TICK_EVENT) = 0;

            //
            // Each tick period handle wired mouse and keyboard.
            //
            if(HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) == 1)
            {
                MouseMoveHandler();
                KeyboardMain();
            }
        }

        //
        // Check for LPRF tick events.  LPRF Ticks are slower since UART to
        // RNP is much slower data connection than the USB.
        //
        if(HWREGBITW(&g_ui32Events, LPRF_TICK_EVENT) == 1)
        {
            //
            // Clear the event flag.
            //
            HWREGBITW(&g_ui32Events, LPRF_TICK_EVENT) = 0;

            //
            // Perform the LPRF Main task handling
            //
            LPRFMain();

        }

        //
        // Check for and handle motion events.
        //
        if((HWREGBITW(&g_ui32Events, MOTION_EVENT) == 1) ||
           (HWREGBITW(&g_ui32Events, MOTION_ERROR_EVENT) == 1))
        {
            //
            // Clear the motion event flag. Set in the Motion I2C interrupt
            // handler when an I2C transaction to get sensor data is complete.
            //
            HWREGBITW(&g_ui32Events, MOTION_EVENT) = 0;

            //
            // Process the motion data that has been captured
            //
            MotionMain();
        }
    }
}
