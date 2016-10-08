//*****************************************************************************
//
// usb_keyb_structs.c - Data structures defining the USB keyboard device.
//
// Copyright (c) 2008-2016 Texas Instruments Incorporated.  All rights reserved.
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
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "usblib/usb-ids.h"
#include "usblib/device/usbdevice.h"
#include "usblib/device/usbdcomp.h"
#include "usblib/device/usbdhid.h"
#include "usblib/device/usbdhidgamepad.h"
#include "usb_cgamepad_structs.h"

//****************************************************************************
//
// The languages supported by this device.
//
//****************************************************************************
const uint8_t g_pui8LangDescriptor[] =
{
    4,
    USB_DTYPE_STRING,
    USBShort(USB_LANG_EN_US)
};

//****************************************************************************
//
// The manufacturer string.
//
//****************************************************************************
const uint8_t g_pui8ManufacturerString[] =
{
    (17 + 1) * 2,
    USB_DTYPE_STRING,
    'T', 0, 'e', 0, 'x', 0, 'a', 0, 's', 0, ' ', 0, 'I', 0, 'n', 0, 's', 0,
    't', 0, 'r', 0, 'u', 0, 'm', 0, 'e', 0, 'n', 0, 't', 0, 's', 0,
};

//****************************************************************************
//
// The product string.
//
//****************************************************************************
const uint8_t g_pui8ProductString[] =
{
    (36 + 1) * 2,
    USB_DTYPE_STRING,
    'T', 0, 'e', 0, 'x', 0, 'a', 0, 's', 0, ' ', 0, 'I', 0, 'n', 0, 's', 0,
    't', 0, 'r', 0, 'u', 0, 'm', 0, 'e', 0, 'n', 0, 't', 0, 's', 0, ' ', 0,
    'T', 0, 'i', 0, 'v', 0, 'a', 0, ' ', 0, 'D', 0, 'u', 0, 'a', 0, 'l', 0,
    ' ', 0, 'G', 0, 'a', 0, 'm', 0, 'e', 0, ' ', 0, 'P', 0, 'a', 0, 'd', 0,
};

//****************************************************************************
//
// The serial number string.
//
//****************************************************************************
const uint8_t g_pui8SerialNumberString[] =
{
    (8 + 1) * 2,
    USB_DTYPE_STRING,
    '1', 0, '2', 0, '3', 0, '4', 0, '5', 0, '6', 0, '7', 0, '8', 0
};

//*****************************************************************************
//
// The interface description string.
//
//*****************************************************************************
const uint8_t g_pui8HIDInterfaceString[] =
{
    (22 + 1) * 2,
    USB_DTYPE_STRING,
    'H', 0, 'I', 0, 'D', 0, ' ', 0, 'G', 0, 'a', 0, 'm', 0, 'e', 0,
    ' ', 0, 'p', 0, 'a', 0, 'd', 0, ' ', 0, 'I', 0, 'n', 0, 't', 0,
    'e', 0, 'r', 0, 'f', 0, 'a', 0, 'c', 0, 'e', 0
};

//*****************************************************************************
//
// The configuration description string.
//
//*****************************************************************************
const uint8_t g_pui8ConfigString[] =
{
    (26 + 1) * 2,
    USB_DTYPE_STRING,
    'H', 0, 'I', 0, 'D', 0, ' ', 0, 'G', 0, 'a', 0, 'm', 0, 'e', 0,
    ' ', 0, 'p', 0, 'a', 0, 'd', 0, ' ', 0, 'C', 0, 'o', 0, 'n', 0,
    'f', 0, 'i', 0, 'g', 0, 'u', 0, 'r', 0, 'a', 0, 't', 0, 'i', 0,
    'o', 0, 'n', 0
};

//*****************************************************************************
//
// The descriptor string table.
//
//*****************************************************************************
const uint8_t * const g_ppui8StringDescriptors[] =
{
    g_pui8LangDescriptor,
    g_pui8ManufacturerString,
    g_pui8ProductString,
    g_pui8SerialNumberString,
    g_pui8HIDInterfaceString,
    g_pui8ConfigString
};

#define NUM_STRING_DESCRIPTORS (sizeof(g_ppui8StringDescriptors) /            \
                                sizeof(uint8_t *))

//*****************************************************************************
//
// The following is the HID report structure definition that is passed back
// to the host.
//
//*****************************************************************************
static const uint8_t g_pui8GameReportDescriptor[] =
{
    UsagePage(USB_HID_GENERIC_DESKTOP),
    Usage(USB_HID_JOYSTICK),
    Collection(USB_HID_APPLICATION),

        //
        // The axis for the controller.
        //
        UsagePage(USB_HID_GENERIC_DESKTOP),
        Usage (USB_HID_POINTER),
        Collection (USB_HID_PHYSICAL),

            //
            // The X, Y, RX, and RY values which are specified as 8-bit
            // absolute position values.
            //
            Usage(USB_HID_X),

            //
            // 4 8-bit absolute values.
            //
            ReportSize(10),
            ReportCount(1),
            Input(USB_HID_INPUT_DATA | USB_HID_INPUT_VARIABLE |
                  USB_HID_INPUT_ABS),

            ReportCount(1),
            ReportSize(6),
            Input(USB_HID_INPUT_CONSTANT),

            Usage (USB_HID_Y),
            ReportSize(10),
            ReportCount(1),
            Input(USB_HID_INPUT_DATA | USB_HID_INPUT_VARIABLE |
                  USB_HID_INPUT_ABS),

            ReportCount(1),
            ReportSize(6),
            Input(USB_HID_INPUT_CONSTANT),

            Usage (USB_HID_Z),
            ReportSize(10),
            ReportCount(1),
            Input(USB_HID_INPUT_DATA | USB_HID_INPUT_VARIABLE |
                  USB_HID_INPUT_ABS),

            ReportCount(1),
            ReportSize(6),
            Input(USB_HID_INPUT_CONSTANT),

            Usage (USB_HID_RX),
            ReportSize(10),
            ReportCount(1),
            Input(USB_HID_INPUT_DATA | USB_HID_INPUT_VARIABLE |
                  USB_HID_INPUT_ABS),

            ReportCount(1),
            ReportSize(6),
            Input(USB_HID_INPUT_CONSTANT),

            //
            // The 16 buttons.
            //
            UsagePage(USB_HID_BUTTONS),
            UsageMinimum(1),
            UsageMaximum(16),
            LogicalMinimum(0),
            LogicalMaximum(1),
            PhysicalMinimum(0),
            PhysicalMaximum(1),

            //
            // 16 - 1 bit values for the buttons.
            //
            ReportSize(1),
            ReportCount(16),
            Input(USB_HID_INPUT_DATA | USB_HID_INPUT_VARIABLE |
                  USB_HID_INPUT_ABS),

        EndCollection,
    EndCollection
};

//*****************************************************************************
//
// The HID game pad device initialization and customization structures.
//
//*****************************************************************************
tUSBDHIDGamepadDevice g_sGamepadDeviceA =
{
    USB_VID_TI_1CBE,
    USB_PID_GAMEPAD,
    0,
    USB_CONF_ATTR_SELF_PWR | USB_CONF_ATTR_RWAKE,
    GamepadHandler,
    (void *)&g_sGamepadDeviceA,
    g_ppui8StringDescriptors,
    NUM_STRING_DESCRIPTORS,
    g_pui8GameReportDescriptor,
    sizeof(g_pui8GameReportDescriptor)
};

//*****************************************************************************
//
// The HID game pad device initialization and customization structures.
//
//*****************************************************************************
tUSBDHIDGamepadDevice g_sGamepadDeviceB =
{
    USB_VID_TI_1CBE,
    USB_PID_GAMEPAD,
    0,
    USB_CONF_ATTR_SELF_PWR | USB_CONF_ATTR_RWAKE,
    GamepadHandler,
    (void *)&g_sGamepadDeviceB,
    g_ppui8StringDescriptors,
    NUM_STRING_DESCRIPTORS,
    g_pui8GameReportDescriptor,
    sizeof(g_pui8GameReportDescriptor)
};


//****************************************************************************
//
// The array of devices supported by this composite device.
//
//****************************************************************************
tCompositeEntry g_psCompDevices[NUM_DEVICES];

//****************************************************************************
//
// Allocate the Device Data for the top level composite device class.
//
//****************************************************************************
tUSBDCompositeDevice g_sCompGameDevice =
{
    //
    // Tiva VID.
    //
    USB_VID_TI_1CBE,

    //
    // Tiva PID for the launch pad based composite game pad device.
    //
    USB_PID_LP_CGAMEPAD,

    //
    // This is in 2mA increments so 500mA.
    //
    0,

    //
    // Bus powered device.
    //
    USB_CONF_ATTR_SELF_PWR,

    //
    // There is no need for a default composite event handler.
    //
    0,

    //
    // The string table.
    //
    g_ppui8StringDescriptors,
    NUM_STRING_DESCRIPTORS,

    //
    // The Composite device array.
    //
    NUM_DEVICES,
    g_psCompDevices,
};

