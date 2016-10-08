//*****************************************************************************
//
// airmouse.c - Main routines for SensHub Air Mouse Demo.
//
// Copyright (c) 2014-2016 Texas Instruments Incorporated.  All rights reserved.
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
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "grlib/grlib.h"
#include "drivers/buttons.h"
#include "drivers/frame.h"
#include "drivers/kentec320x240x16_ssd2119.h"
#include "drivers/pinout.h"
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "usblib/device/usbdevice.h"
#include "usblib/device/usbdcomp.h"
#include "usblib/device/usbdhid.h"
#include "usblib/device/usbdhidmouse.h"
#include "usblib/device/usbdhidkeyb.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "remoti_uart.h"
#include "remoti_npi.h"
#include "remoti_rti.h"
#include "remoti_rtis.h"
#include "events.h"
#include "motion.h"
#include "usb_structs.h"
#include "lprf.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Motion Air Mouse (airmouse)</h1>
//!
//! This example demonstrates the use of the Sensor Library, DK-TM4C129X and
//! the SensHub BoosterPack to fuse nine axis sensor measurements into motion
//! and gesture events.  These events are then transformed into mouse and
//! keyboard events to perform standard HID tasks.
//!
//! Connect the USB OTG port, between the EM connectors of the DK-TM4C129X, to
//! a standard computer USB port.  The DK-TM4C129X with SensHub BoosterPack
//! enumerates on the USB bus as a composite HID keyboard and mouse.
//!
//! Hold the DK-TM4C129X with the buttons and LCD away from the user and toward
//! the computer with USB Device cable exiting from the left side of the board.
//!
//! - Roll or tilt the DK-TM4C129X to move the mouse cursor of the computer
//! up, down, left and right.
//!
//! - The buttons on the DK-TM4C129X, SEL and DOWN, perform the left and right
//! mouse click actions respectively.  The buttons on the SensHub BoosterPack
//! are not currently used by this example.
//!
//! - A quick spin of the DK-TM4C129X generates a PAGE_UP or PAGE_DOWN keyboard
//! press and release depending on the direction of the spin.  This motion
//! simulates scrolling.
//!
//! - A quick horizontal jerk to the left or right  generates a CTRL+ or CTRL-
//! keyboard event, which creates the zoom effect used in many applications,
//! especially web browsers.
//!
//! - A quick vertical lift generates an ALT+TAB keyboard event, which allows
//! the computer user to select between currently open windows.
//!
//! - A quick twist to the left or right moves the window selector.
//!
//! - A quick jerk in the down direction selects the desired window and
//! closes the window selection dialog.
//!
//! This example also supports the RemoTI low power RF Zigbee&reg;&nbsp;human
//! interface device profile.  The wireless features of this example require
//! the CC2533EMK expansion card and the CC2531EMK USB Dongle.  For details and
//! instructions for wireless operations see the Wiki at
//! http://processors.wiki.ti.com/index.php/Wireless_Air_Mouse_Guide.
//
//*****************************************************************************

//*****************************************************************************
//
// Holds the system clock.
//
//*****************************************************************************
uint32_t g_ui32SysClock;

//*****************************************************************************
//
// Structure to hold the graphics context.
//
//*****************************************************************************
tContext g_sContext;

//*****************************************************************************
//
// Structures to hold a rectangular area to print status and user instructions.
//
//*****************************************************************************
tRectangle g_sUserInfoRect;
tRectangle g_sUSBStatsRect;
tRectangle g_sLPRFStatsRect;

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
// This is the main loop that runs the application.
//
//*****************************************************************************
int
main(void)
{
    uint32_t ui32PLLRate;

    //
    // Set the clocking to run from the PLL at 120MHz.
    //
    g_ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                             SYSCTL_OSC_MAIN |
                                             SYSCTL_USE_PLL |
                                             SYSCTL_CFG_VCO_480), 120000000);

    //
    // Set the system tick to fire 100 times per second.
    //
    MAP_SysTickPeriodSet(g_ui32SysClock / SYSTICKS_PER_SECOND);
    MAP_SysTickIntEnable();
    MAP_SysTickEnable();

    //
    // Configure the device pins.
    //
    PinoutSet();

    //
    // Initialize the display driver.
    //
    Kentec320x240x16_SSD2119Init(g_ui32SysClock);

    //
    // Initialize the graphics context.
    //
    GrContextInit(&g_sContext, &g_sKentec320x240x16_SSD2119);

    //
    // Draw the application frame.
    //
    FrameDraw(&g_sContext, "air mouse app");

    //
    // Flush any cached drawing operations.
    //
    GrFlush(&g_sContext);

    //
    // Set font size for rest of the application.
    //
    GrContextFontSet(&g_sContext, g_psFontCm18b);

    //
    // Configure a region on the LCD to display user instructions.
    //
    g_sUserInfoRect.i16XMin = 10;
    g_sUserInfoRect.i16YMin = 140;
    g_sUserInfoRect.i16XMax = 310;
    g_sUserInfoRect.i16YMax = 220;

    //
    // Configure a region on the LCD to display USB connection status.
    //
    g_sUSBStatsRect.i16XMin = 120;
    g_sUSBStatsRect.i16YMin = 65;
    g_sUSBStatsRect.i16XMax = 310;
    g_sUSBStatsRect.i16YMax = 89;

    //
    // Configure a region on the LCD to display LPRF connection status.
    //
    g_sLPRFStatsRect.i16XMin = 120;
    g_sLPRFStatsRect.i16YMin = 90;
    g_sLPRFStatsRect.i16XMax = 310;
    g_sLPRFStatsRect.i16YMax = 114;

    //
    // Enable UART0
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, g_ui32SysClock);

    //
    // Print the welcome message to the terminal.
    //
    UARTprintf("\033[2JAir Mouse Application\n");

    //
    // Print the default USB and Zigbee status on LCD.
    //
    GrStringDraw(&g_sContext, "USB", -1, 60, 65, 1);
    GrStringDraw(&g_sContext, "Zigbee", -1, 60, 90, 1);
    GrContextForegroundSet(&g_sContext, ClrRed);
    GrStringDraw(&g_sContext, "Disconnected", -1, 120, 65, 1);
    GrStringDraw(&g_sContext, "Disconnected", -1, 120, 90, 1);

    //
    // Display the default usage instructions in gray.
    //
    GrContextForegroundSet(&g_sContext, ClrGray);
    GrContextFontSet(&g_sContext, g_psFontCmss16b);
    GrStringDraw(&g_sContext, "Connect DK-TM4C129X's USB OTG", -1, 10, 140, 1);
    GrStringDraw(&g_sContext, "connector to a PC to use as wired mouse", -1,
                 10, 160, 1);
    GrStringDraw(&g_sContext, "or refer Readme to use as an air mouse.", -1,
                 10, 180, 1);
    GrContextForegroundSet(&g_sContext, ClrWhite);
    GrContextFontSet(&g_sContext, g_psFontCm18b);

    //
    // Configure desired interrupt priorities. This makes certain that the DCM
    // is fed data at a consistent rate. Lower numbers equal higher priority.
    //
    MAP_IntPrioritySet(INT_I2C3, 0x00);
    MAP_IntPrioritySet(INT_GPIOS, 0x10);
    MAP_IntPrioritySet(FAULT_SYSTICK, 0x20);
    MAP_IntPrioritySet(INT_UART5, 0x60);
    MAP_IntPrioritySet(INT_UART0, 0x70);

    //
    // Configure the USB D+ and D- pins.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOL);
    MAP_GPIOPinTypeUSBAnalog(GPIO_PORTL_BASE, GPIO_PIN_6 | GPIO_PIN_7);

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
    // Tell the USB library the CPU clock and the PLL frequency.  This is a
    // new requirement for TM4C129 devices.
    //
    SysCtlVCOGet(SYSCTL_XTAL_25MHZ, &ui32PLLRate);
    USBDCDFeatureSet(0, USBLIB_FEATURE_CPUCLK, &g_ui32SysClock);
    USBDCDFeatureSet(0, USBLIB_FEATURE_USBPLL, &ui32PLLRate);

    //
    // Pass the device information to the USB library and place the device
    // on the bus.
    //
    USBDCompositeInit(0, &g_sCompDevice, DESCRIPTOR_DATA_SIZE,
                      g_pui8DescriptorData);

    //
    // Initialize SEL and DOWN buttons to work as left and right click of the
    // mouse.
    //
    ButtonsInit(SELECT_BUTTON | DOWN_BUTTON);

    //
    // Initialize the motion sub system.
    //
    MotionInit();

    //
    // Initialize the Radio Systems.
    //
    LPRFInit();

    //
    // Configure PQ4 and PN5 to control the blue and red LED.
    //
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTQ_BASE, GPIO_PIN_4);
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTN_BASE, GPIO_PIN_5);
    MAP_GPIOPinWrite(GPIO_PORTQ_BASE, GPIO_PIN_4, 0);
    MAP_GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_5, 0);

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
