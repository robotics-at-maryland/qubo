//****************************************************************************
//
// usb_dev_comp_hid.c - Routines for handling the mouse.
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
//****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/rom_map.h"
#include "driverlib/rom.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "grlib/grlib.h"
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "usblib/device/usbdevice.h"
#include "usblib/device/usbdcomp.h"
#include "usblib/device/usbdhid.h"
#include "usblib/device/usbdhidmouse.h"
#include "usblib/device/usbdhidkeyb.h"
#include "drivers/buttons.h"
#include "usb_structs.h"
#include "events.h"
#include "motion.h"

//*****************************************************************************
//
// Global variable that holds clock frequency.
//
//*****************************************************************************
extern uint32_t g_ui32SysClock;

//*****************************************************************************
//
// Structure to hold the graphics context.
//
//*****************************************************************************
extern tContext g_sContext;

//*****************************************************************************
//
// Structures to hold a rectangular area to print status and user instructions.
//
//*****************************************************************************
extern tRectangle g_sUserInfoRect;
extern tRectangle g_sUSBStatsRect;

//*****************************************************************************
//
// Different USB display status used by g_ui8PrevUSBDisplay.
//
//*****************************************************************************
enum
{
    DISPLAY_USB_NOT_CONNECTED = 0,
    DISPLAY_USB_CONNECTED
};

//*****************************************************************************
//
// Global storage for previously displayed USB status.  This is needed to avoid
// printing the same status repeatedly.
//
//*****************************************************************************
uint_fast8_t g_ui8PrevUSBDisplay = DISPLAY_USB_NOT_CONNECTED;

//*****************************************************************************
//
// State of the buttons on previous send of a mouse packet.
//
//*****************************************************************************
uint_fast8_t g_ui8MouseButtonsPrev;

//****************************************************************************
//
// The number of system ticks to wait for each USB packet to be sent before
// we assume the host has disconnected.  The value 50 equates to half a
// second.
//
//****************************************************************************
#define MAX_SEND_DELAY          50

//****************************************************************************
//
// This enumeration holds the various states that the mouse can be in during
// normal operation.
//
//****************************************************************************
volatile enum
{
    //
    // Not configured.
    //
    MOUSE_STATE_UNCONFIGURED,

    //
    // No keys to send and not waiting on data.
    //
    MOUSE_STATE_IDLE,

    //
    // Waiting on data to be sent out.
    //
    MOUSE_STATE_SENDING
}
g_eMouseState = MOUSE_STATE_UNCONFIGURED;

//*****************************************************************************
//
// This enumeration holds the various states that the keyboard can be in during
// normal operation.
//
//*****************************************************************************
volatile enum
{
    //
    // Unconfigured.
    //
    KEYBOARD_STATE_UNCONFIGURED,

    //
    // No keys to send and not waiting on data.
    //
    KEYBOARD_STATE_IDLE,

    //
    // Waiting on data to be sent out.
    //
    KEYBOARD_STATE_SENDING
}
g_eKeyboardState = KEYBOARD_STATE_UNCONFIGURED;

//****************************************************************************
//
// This function handles notification messages from the mouse device driver.
//
//****************************************************************************
uint32_t
MouseHandler(void *pvCBData, uint32_t ui32Event,
             uint32_t ui32MsgData, void *pvMsgData)
{
    switch(ui32Event)
    {
        //
        // The USB host has connected to and configured the device.
        //
        case USB_EVENT_CONNECTED:
        {
            g_eMouseState = MOUSE_STATE_IDLE;
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 1;

            //
            // Display on LCD to show we are connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrGreen);
                GrStringDraw(&g_sContext, "Connected", -1, 120, 65, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);

                //
                // Clear usage instructions as nothing to inform here.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
            }
            break;
        }

        //
        // The USB host has disconnected from the device.
        //
        case USB_EVENT_DISCONNECTED:
        {
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 0;
            g_eMouseState = MOUSE_STATE_UNCONFIGURED;

            //
            // Display on LCD to show we are no longer connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_NOT_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_NOT_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrRed);
                GrStringDraw(&g_sContext, "Disconnected", -1, 120, 65, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);

                //
                // Clear usage instructions as nothing to inform.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
            }
            break;
        }

        //
        // A report was sent to the host.  We are not free to send another.
        //
        case USB_EVENT_TX_COMPLETE:
        {
            g_eMouseState = MOUSE_STATE_IDLE;
            break;
        }
    }

    return(0);
}

//***************************************************************************
//
// Wait for a period of time for the state to become idle.
//
// \param ui32TimeoutTick is the number of system ticks to wait before
// declaring a timeout and returning \b false.
//
// This function polls the current keyboard state for ui32TimeoutTicks system
// ticks waiting for it to become idle.  If the state becomes idle, the
// function returns true.  If it ui32TimeoutTicks occur prior to the state
// becoming idle, false is returned to indicate a timeout.
//
// \return Returns \b true on success or \b false on timeout.
//
//***************************************************************************
bool
MouseWaitForSendIdle(uint32_t ui32TimeoutTicks)
{
    uint32_t ui32Start;
    uint32_t ui32Now;
    uint32_t ui32Elapsed;

    ui32Start = g_ui32SysTickCount;
    ui32Elapsed = 0;

    while(ui32Elapsed < ui32TimeoutTicks)
    {
        //
        // Is the mouse is idle, return immediately.
        //
        if(g_eMouseState == MOUSE_STATE_IDLE)
        {
            return(true);
        }

        //
        // Determine how much time has elapsed since we started waiting.  This
        // should be safe across a wrap of g_ui32SysTickCount.
        //
        ui32Now = g_ui32SysTickCount;
        ui32Elapsed = ((ui32Start < ui32Now) ? (ui32Now - ui32Start) :
                     (((uint32_t)0xFFFFFFFF - ui32Start) + ui32Now + 1));
    }

    //
    // If we get here, we timed out so return a bad return code to let the
    // caller know.
    //
    return(false);
}

//****************************************************************************
//
// This function provides simulated movements of the mouse.
//
//****************************************************************************
void
MouseMoveHandler(void)
{
    uint32_t ui32Retcode;
    int8_t i8DeltaX, i8DeltaY;
    uint8_t ui8Buttons;
    bool bSendIt, bButtonChange;

    //
    // Get the current motion mouse behaviors.
    //
    bSendIt = MotionMouseGet(&i8DeltaX, &i8DeltaY, &ui8Buttons);

    if((g_ui8Buttons & ALL_BUTTONS) != (g_ui8MouseButtonsPrev))
    {
        bButtonChange = true;
        g_ui8MouseButtonsPrev = g_ui8Buttons;
    }
    else
    {
        bButtonChange = false;
    }

    if(bSendIt || bButtonChange)
    {
        //
        // Convert button presses from GPIO pin positions to mouse button bit
        // positions. Overrides Motion Button action (currently not
        // implemented).
        //
        ui8Buttons = (g_ui8Buttons & DOWN_BUTTON) >> 4;
        ui8Buttons |= (g_ui8Buttons & SELECT_BUTTON) >> 1;

        //
        // Tell the HID driver to send this new report.
        //
        g_eMouseState = MOUSE_STATE_SENDING;
        ui32Retcode = USBDHIDMouseStateChange((void *)&g_sMouseDevice,
                                              i8DeltaX, i8DeltaY, ui8Buttons);

        //
        // Did we schedule the report for transmission?
        //
        if(ui32Retcode == MOUSE_SUCCESS)
        {
            //
            // Wait for the host to acknowledge the transmission if all went
            // well.
            //
            if(!MouseWaitForSendIdle(MAX_SEND_DELAY))
            {
                //
                // The transmission failed, so assume the host disconnected and
                // go back to waiting for a new connection.
                //
                HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 0;

                //
                // Display on LCD to show we are no longer connected to USB.
                //
                if(g_ui8PrevUSBDisplay != DISPLAY_USB_NOT_CONNECTED)
                {
                    g_ui8PrevUSBDisplay = DISPLAY_USB_NOT_CONNECTED;
                    DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect,
                                ClrBlack);
                    GrContextForegroundSet(&g_sContext, ClrRed);
                    GrStringDraw(&g_sContext, "Disconnected", -1, 120, 65, 1);
                    GrContextForegroundSet(&g_sContext, ClrWhite);

                    //
                    // Clear usage instructions as nothing to inform here.
                    //
                    DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect,
                                ClrBlack);
                }
            }
        }
    }
}

//*****************************************************************************
//
// Handles asynchronous events from the HID keyboard driver.
//
// \param pvCBData is the event callback pointer provided during
// USBDHIDKeyboardInit().  This is a pointer to our keyboard device structure
// (&g_sKeyboardDevice).
// \param ui32Event identifies the event we are being called back for.
// \param ui32MsgData is an event-specific value.
// \param pvMsgData is an event-specific pointer.
//
// This function is called by the HID keyboard driver to inform the application
// of particular asynchronous events related to operation of the keyboard HID
// device.
//
// \return Returns 0 in all cases.
//
//*****************************************************************************
uint32_t
KeyboardHandler(void *pvCBData, uint32_t ui32Event, uint32_t ui32MsgData,
                void *pvMsgData)
{
    switch (ui32Event)
    {
        //
        // The host has connected to us and configured the device.
        //
        case USB_EVENT_CONNECTED:
        {
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 1;
            HWREGBITW(&g_ui32USBFlags, FLAG_SUSPENDED) = 0;
            g_eKeyboardState = KEYBOARD_STATE_IDLE;

            //
            // Display on LCD to show we are connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrGreen);
                GrStringDraw(&g_sContext, "Connected", -1, 120, 65, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);

                //
                // Clear usage instructions as nothing to inform.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
            }
            break;
        }

        //
        // The host has disconnected from us.
        //
        case USB_EVENT_DISCONNECTED:
        {
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 0;
            g_eKeyboardState = KEYBOARD_STATE_UNCONFIGURED;

            //
            // Display on LCD to show we are no longer connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_NOT_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_NOT_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrRed);
                GrStringDraw(&g_sContext, "Disconnected", -1, 120, 65, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);

                //
                // Clear usage instructions as nothing to inform.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
            }
            break;
        }

        //
        // We receive this event every time the host acknowledges transmission
        // of a report. It is used here purely as a way of determining whether
        // the host is still talking to us or not.
        //
        case USB_EVENT_TX_COMPLETE:
        {
            //
            // Enter the idle state since we finished sending something.
            //
            g_eKeyboardState = KEYBOARD_STATE_IDLE;
            break;
        }

        //
        // This event indicates that the host has suspended the USB bus.
        //
        case USB_EVENT_SUSPEND:
        {
            HWREGBITW(&g_ui32USBFlags, FLAG_SUSPENDED) = 1;
            break;
        }

        //
        // This event signals that the host has resumed signaling on the bus.
        //
        case USB_EVENT_RESUME:
        {
            HWREGBITW(&g_ui32USBFlags, FLAG_SUSPENDED) = 0;
            break;
        }

        //
        // This event indicates that the host has sent us an Output or
        // Feature report and that the report is now in the buffer we provided
        // on the previous USBD_HID_EVENT_GET_REPORT_BUFFER callback.
        //
        case USBD_HID_KEYB_EVENT_SET_LEDS:
        {
            break;
        }

        //
        // We ignore all other events.
        //
        default:
        {
            break;
        }
    }

    return(0);
}

//***************************************************************************
//
// Wait for a period of time for the state to become idle.
//
// \param ui32TimeoutTick is the number of system ticks to wait before
// declaring a timeout and returning \b false.
//
// This function polls the current keyboard state for ui32TimeoutTicks system
// ticks waiting for it to become idle.  If the state becomes idle, the
// function returns true.  If it ui32TimeoutTicks occur prior to the state
// becoming idle, false is returned to indicate a timeout.
//
// \return Returns \b true on success or \b false on timeout.
//
//***************************************************************************
bool
KeyboardWaitForSendIdle(uint_fast32_t ui32TimeoutTicks)
{
    uint32_t ui32Start;
    uint32_t ui32Now;
    uint32_t ui32Elapsed;

    ui32Start = g_ui32SysTickCount;
    ui32Elapsed = 0;

    while(ui32Elapsed < ui32TimeoutTicks)
    {
        //
        // Is the keyboard is idle, return immediately.
        //
        if(g_eKeyboardState == KEYBOARD_STATE_IDLE)
        {
            return(true);
        }

        //
        // Determine how much time has elapsed since we started waiting.  This
        // should be safe across a wrap of g_ui32SysTickCount.
        //
        ui32Now = g_ui32SysTickCount;
        ui32Elapsed = ((ui32Start < ui32Now) ? (ui32Now - ui32Start) :
                     (((uint32_t)0xFFFFFFFF - ui32Start) + ui32Now + 1));
    }

    //
    // If we get here, we timed out so return a bad return code to let the
    // caller know.
    //
    return(false);
}

//****************************************************************************
//
// This function will send a KeyState change to the USB library and wait for
// the transmission to complete.
//
//****************************************************************************
uint32_t KeyboardStateChange(uint8_t ui8Modifiers, uint8_t ui8Usage,
                             bool bPressed)
{
    uint32_t ui32RetCode;

    //
    // Send the Key state change to the
    //
    g_eKeyboardState = KEYBOARD_STATE_SENDING;
    ui32RetCode = USBDHIDKeyboardKeyStateChange((void *)&g_sKeyboardDevice,
                                                ui8Modifiers, ui8Usage,
                                                bPressed);

    if(ui32RetCode == KEYB_SUCCESS)
    {
        //
        // Wait until the key press message has been sent.
        //
        if(!KeyboardWaitForSendIdle(MAX_SEND_DELAY))
        {
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 0;

            //
            // Display on LCD to show we are no longer connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_NOT_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_NOT_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrRed);
                GrStringDraw(&g_sContext, "Disconnected", -1, 120, 65, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);

                //
                // Clear usage instructions as nothing to inform.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
            }
        }
    }

    return (ui32RetCode);
}

//****************************************************************************
//
// Primary keyboard application function.  This function translates the Gesture
// states provided by the motion system into keyboard events to emulate certain
// application functions.
//
// A quick lift will simulate an ALT + TAB. While lifted a twist left or right
// will select amongst the open windows presented in the ALT + TAB dialog. A
// quick down motion will return to the idle state and ready for mousing.
//
// While in the idle state a quick twist about the Z axis will page up or page
// down the current window depending on direction of rotation.  A sharp
// horizontal left or right acceleration will send a CTRL + or CTRL - depending
// on direction.
//
// Roll and Pitch while idle move the mouse cursor.  See MouseMoveHandler.
//
//****************************************************************************
void
KeyboardMain(void)
{
    bool bKeyHold, bModifierHold;
    uint8_t ui8Key, ui8Modifiers;

    //
    // Check if the keyboard is in the suspended state.
    //
    if(HWREGBITW(&g_ui32USBFlags, FLAG_SUSPENDED) == 1)
    {
        //
        // We are connected but keyboard is suspended so do wake request.
        //
        USBDHIDKeyboardRemoteWakeupRequest((void *)&g_sKeyboardDevice);
    }
    else
    {
        if(MotionKeyboardGet(&ui8Modifiers, &ui8Key, &bModifierHold,
                             &bKeyHold))
        {
            //
            // Send the Keyboard Packet button presses.
            //
            if(KeyboardStateChange(ui8Modifiers, ui8Key, true) != KEYB_SUCCESS)
            {
                return;
            }

            //
            // Release the key and modifier depending on the hold state that
            // was returned.
            //
            ROM_SysCtlDelay(g_ui32SysClock / (100 * 3));
            if(KeyboardStateChange(bModifierHold ? ui8Modifiers : 0,
                                   ui8Key, bKeyHold) != KEYB_SUCCESS)
            {
                return;
            }
        }
    }
}

//****************************************************************************
//
// Generic event handler for the composite device. This should handle events
// that come in for endpoint zero of our composite device.
//
//****************************************************************************
uint32_t
EventHandler(void *pvCBData, uint32_t ui32Event, uint32_t ui32MsgData,
             void *pvMsgData)
{
    switch(ui32Event)
    {
        //
        // The host has connected to us and configured the device.
        //
        case USB_EVENT_CONNECTED:
        {
            //
            // Now connected.
            //
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 1;

            //
            // Display on LCD to show we are connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrGreen);
                GrStringDraw(&g_sContext, "Connected", -1, 120, 65, 1);

                //
                // Display usage instructions in gray.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrGray);
                GrContextFontSet(&g_sContext, g_psFontCmss16b);
                GrStringDraw(&g_sContext,
                             "Hold DK-TM4C129X so that the buttons", -1, 10,
                             140, 1);
                GrStringDraw(&g_sContext,
                             "and LCD are away from the user. Start", -1, 10,
                             160, 1);
                GrStringDraw(&g_sContext,
                             "using DK-TM4C129X as a wired mouse.", -1, 10,
                             180, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);
                GrContextFontSet(&g_sContext, g_psFontCm18b);
            }
            break;
        }

        //
        // Handle the disconnect state.
        //
        case USB_EVENT_DISCONNECTED:
        {
            //
            // No longer connected.
            //
            HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) = 0;

            //
            // Display on LCD to show we are no longer connected to USB.
            //
            if(g_ui8PrevUSBDisplay != DISPLAY_USB_NOT_CONNECTED)
            {
                g_ui8PrevUSBDisplay = DISPLAY_USB_NOT_CONNECTED;
                DpyRectFill(g_sContext.psDisplay, &g_sUSBStatsRect, ClrBlack);
                GrContextForegroundSet(&g_sContext, ClrRed);
                GrStringDraw(&g_sContext, "Disconnected", -1, 120, 65, 1);
                GrContextForegroundSet(&g_sContext, ClrWhite);

                //
                // Clear usage instructions as nothing to inform.
                //
                DpyRectFill(g_sContext.psDisplay, &g_sUserInfoRect, ClrBlack);
            }
            break;
        }

        default:
        {
            break;
        }
    }

    return(0);
}
