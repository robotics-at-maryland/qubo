//*****************************************************************************
//
// remoti_zid.h - Defines and typedefs for Zigbee human interface device.
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

#ifndef __REMOTI_ZID_H__
#define __REMOTI_ZID_H__

//*****************************************************************************
//
// If building with a C++ compiler, make all of the definitions in this header
// have a C binding.
//
//*****************************************************************************
#ifdef __cplusplus
extern "C"
{
#endif

//*****************************************************************************
//
// Zigbee profile version number.
//
//*****************************************************************************
#define ZID_PROFILE_VERSION     0x0100

//*****************************************************************************
//
// Zigbee indexes.
//
//*****************************************************************************
#define ZID_FRAME_CTL_IDX       0
#define ZID_DATA_BUF_IDX        1

//*****************************************************************************
//
// ZID  Command Code field values (Table 2).
//
//*****************************************************************************
#define ZID_CMD_GET_REPORT      0x01
#define ZID_CMD_REPORT_DATA     0x02
#define ZID_CMD_SET_REPORT      0x03
#define ZID_CMD_CODE_MASK       0x4F

//*****************************************************************************
//
// ZID response code field (table 3)
//
//*****************************************************************************
#define ZID_GENERIC_RSP_INVALID_REPORT_ID                                     \
                                0x40
#define ZID_GENERIC_RSP_MISSING_FRAGMENT                                      \
                                0x41

//*****************************************************************************
//
// ZID report type field (table 4)
//
//*****************************************************************************
#define ZID_REPORT_TYPE_IN      0x01
#define ZID_REPORT_TYPE_OUT     0x02
#define ZID_REPORT_TYPE_FEATURE 0x03

//*****************************************************************************
//
// Current protocol attribute values (table 13)
//
//*****************************************************************************
#define ZID_PROTO_BOOT          0x00
#define ZID_PROTO_REPORT        0x01

//*****************************************************************************
//
// HID descriptor type (table 9)
//
//*****************************************************************************
#define ZID_DESC_TYPE_REPORT    0x22
#define ZID_DESC_TYPE_PHYSICAL  0x23

//*****************************************************************************
//
// Standard RF4CE ZID profile report IDs (table 30)
//
//*****************************************************************************
#define ZID_STD_REPORT_NONE     0x00
#define ZID_STD_REPORT_MOUSE    0x01
#define ZID_STD_REPORT_KEYBOARD 0x02
#define ZID_STD_REPORT_CONTACT_DATA                                           \
                                0x03
#define ZID_STD_REPORT_GESTURE_TAP                                            \
                                0x04
#define ZID_STD_REPORT_GESTURE_SCROLL                                         \
                                0x05
#define ZID_STD_REPORT_GESTURE_PINCH                                          \
                                0x06
#define ZID_STD_REPORT_GESTURE_ROTATE                                         \
                                0x07
#define ZID_STD_REPORT_GESTURE_SYNC                                           \
                                0x08
#define ZID_STD_REPORT_TOUCH_SENSOR_PROPERTIES                                \
                                0x09
#define ZID_STD_REPORT_TAP_SUPPORT_PROPERTIES                                 \
                                0x0A

//*****************************************************************************
//
// Data and report lengths for over the air mouse and keyboard packets.
//
//*****************************************************************************
#define ZID_MOUSE_DATA_LENGTH   3
#define ZID_MOUSE_REPORT_LENGTH (ZID_MOUSE_DATA_LENGTH + 2)
#define ZID_KEYBOARD_DATA_LENGTH                                              \
                                8
#define ZID_KEYBOARD_REPORT_LENGTH                                            \
                                (ZID_KEYBOARD_DATA_LENGTH + 2)

//*****************************************************************************
//
// Transmit Pipe configuration
//
//*****************************************************************************
#if !defined ZID_SECURE_INTERRUPT_PIPES
#define ZID_SECURE_INTERRUPT_PIPES                                            \
                                0
#endif

#if ZID_SECURE_INTERRUPT_PIPES
#define ZID_TX_OPTIONS_INTERRUPT_PIPE                                         \
                                (RTI_TX_OPTION_SINGLE_CHANNEL |               \
                                 RTI_TX_OPTION_SECURITY)
#else
#define ZID_TX_OPTIONS_INTERRUPT_PIPE                                         \
                                (RTI_TX_OPTION_SINGLE_CHANNEL)
#endif

#define ZID_TX_OPTIONS_CONTROL_PIPE                                           \
                                (RTI_TX_OPTION_ACKNOWLEDGED |                 \
                                 RTI_TX_OPTION_SECURITY)

#define ZID_TX_OPTIONS_CONTROL_PIPE_BROADCAST                                 \
                                (RTI_TX_OPTION_ACKNOWLEDGED |                 \
                                 RTI_TX_OPTION_SECURITY |                     \
                                 RTI_TX_OPTION_BROADCAST)

//*****************************************************************************
//
// Zigbee Get Report command structure.
//
//*****************************************************************************
typedef struct
{
    //
    // ZID header command.
    //
    uint8_t ui8Command;

    //
    // Report type.
    //
    uint8_t ui8Type;

    //
    // Report Id.
    //
    uint8_t ui8ID;
} tZIDGetReportCommand;

//*****************************************************************************
//
// Mouse data structure.
//
//*****************************************************************************
typedef struct
{
    //
    // Button State.
    //
    uint8_t ui8Buttons;

    //
    // Cursor change in the X direction.
    //
    int8_t i8DeltaX;

    //
    // Cursor change in the Y direction.
    //
    int8_t i8DeltaY;
} tZIDMouseData;

//*****************************************************************************
//
// Keyboard data structure.
//
//*****************************************************************************
typedef struct
{
    //
    // Modifier keys such as ALT and SHIFT. Also LED states.
    //
    uint8_t ui8Modifiers;

    //
    // Reserved.
    //
    uint8_t ui8Reserved;

    //
    // Array of up to 6 keys currently being pressed.
    //
    unsigned char pui8Keys[6];
} tZIDKeyboardData;

//*****************************************************************************
//
// Zigbee human interface report structure.
//
//*****************************************************************************//
typedef struct
{
    //
    // Length of report.
    //
    uint8_t ui8Length;

    //
    // Report Type.
    //
    uint8_t ui8Type;

    //
    // Report Identifier.
    //
    uint8_t ui8ID;

    //
    // Union for storing either mouse or keyboard data.
    //
    union
    {
        tZIDMouseData sMouseData;
        tZIDKeyboardData sKeyboardData;
    }uData;
} tZIDReportRecord;

//*****************************************************************************
//
// Zigbee report data command structure wrapper.
//
//*****************************************************************************
typedef struct
{
    //
    // ZID header command.
    //
    uint8_t ui8Command;

    //
    // Report Record.
    //
    tZIDReportRecord sReportRecord;
} tZIDReportDataCommand;


//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

//*****************************************************************************
//
// Prototypes for the globals exported by this driver.
//
//*****************************************************************************

#endif // __REMOTI_ZID_H__
