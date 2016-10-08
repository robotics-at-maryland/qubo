//*****************************************************************************
//
// lprf.c - Implementation of the applications low power radio frequency
// interface.
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
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/pin_map.h"
#include "driverlib/gpio.h"
#include "driverlib/sysctl.h"
#include "driverlib/rom.h"
#include "driverlib/interrupt.h"
#include "drivers/rgb.h"
#include "remoti_uart.h"
#include "remoti_npi.h"
#include "remoti_rti.h"
#include "remoti_rtis.h"
#include "remoti_zid.h"
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "usblib/device/usbdevice.h"
#include "usblib/device/usbdcomp.h"
#include "usblib/device/usbdhid.h"
#include "usblib/device/usbdhidmouse.h"
#include "usblib/device/usbdhidkeyb.h"
#include "utils/uartstdio.h"
#include "drivers/buttons.h"
#include "events.h"
#include "motion.h"
#include "usb_structs.h"

//*****************************************************************************
//
// Global array that contains the colors of the RGB.  Low power RF is assigned
// green.
//
// Slow blinking means trying to pair.
// Solid means paired.
// Quick blinks occur when data is transmitted.  (Light off on transmit attempt
// light on when transmit attempt is complete).
// off not paired.
//
//*****************************************************************************
extern volatile uint32_t g_pui32RGBColors[3];

//*****************************************************************************
//
// Global storage to count blink ticks.  This sets blink timing after error.
//
//*****************************************************************************
uint32_t g_ui32RGBLPRFBlinkCounter;

//*****************************************************************************
//
// Global storage for buttons state of previous LPRF packet
//
//*****************************************************************************
uint_fast8_t g_ui8LPRFButtonsPrev;

//*****************************************************************************
//
// Current state of the LPRF network link.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8LinkState;

//*****************************************************************************
//
// The index into our pairing table that contains the current active link.
//
//*****************************************************************************
uint8_t g_ui8LinkDestIndex;

//*****************************************************************************
//
// Link States.
//
//*****************************************************************************
enum
{
  LINK_STATE_INIT,
  LINK_STATE_READY,
  LINK_STATE_PAIR,
  LINK_STATE_NDATA,
  LINK_STATE_UNPAIR,
  LINK_STATE_TEST,
  LINK_STATE_TEST_COMPLETED,
  LINK_STATE_OAD
};

//*****************************************************************************
//
// Global variables to hold key presses and hold information.
//
//*****************************************************************************
tZIDKeyboardData g_sZIDKeys;
bool g_bModifierHold;
bool g_bKeyHold;
bool g_bPressed;

//*****************************************************************************
//
// List of implemented device types.
//
//*****************************************************************************
const uint8_t pui8DevList[RTI_MAX_NUM_DEV_TYPES] =
{
  RTI_DEVICE_REMOTE_CONTROL,
  RTI_DEVICE_RESERVED_INVALID,
  RTI_DEVICE_RESERVED_INVALID
};

//*****************************************************************************
//
// List of implemented profiles.
//
//*****************************************************************************
const uint8_t pui8ProfileList[RTI_MAX_NUM_PROFILE_IDS] =
{
    RTI_PROFILE_ZRC, RTI_PROFILE_ZID, 0, 0, 0, 0, 0
};

//*****************************************************************************
//
// List of possible target types.
//
//*****************************************************************************
const unsigned char pucTargetList[RTI_MAX_NUM_SUPPORTED_TGT_TYPES] =
{
    RTI_DEVICE_TELEVISION,
    RTI_DEVICE_VIDEO_PLAYER_RECORDER,
    RTI_DEVICE_SET_TOP_BOX,
    RTI_DEVICE_MEDIA_CENTER_PC,
    RTI_DEVICE_GAME_CONSOLE,
    RTI_DEVICE_MONITOR
};

//*****************************************************************************
//
// String version of Vendor Name.
//
//*****************************************************************************
const unsigned char pui8VendorName[]="TI-LPRF";

//*****************************************************************************
//
// Determine type of reset to issue based on user buttons presses.
// No buttons will cause a restore of all previous state values.
// Left button press will clear state information but not configuration.
// Right button will clear state information and configuration.
//
// Reset the RNP by toggling the RF modules external reset line.
//
//*****************************************************************************
void
ZIDResetRNP(void)
{
    //
    // Assert reset to the RNP.
    //
    ROM_GPIOPinWrite(GPIO_PORTC_BASE, GPIO_PIN_7, 0);

    //
    // Hold reset low for about 8 milliseconds to verify reset is detected
    //
    ROM_SysCtlDelay(ROM_SysCtlClockGet() / (125 * 3));

    //
    //Release reset to the RNP
    //
    ROM_GPIOPinWrite(GPIO_PORTC_BASE, GPIO_PIN_7, GPIO_PIN_7);

    //
    // Delay to allow RNP to do its internal boot.
    //
    ROM_SysCtlDelay(ROM_SysCtlClockGet() / (2 * 3));
}

//*****************************************************************************
//
// Based on desired restoration setting, configure the RNP parameters for this
// application.
//
//*****************************************************************************
void
ZIDConfigParams(void)
{
    uint8_t pui8Value[MAX_AVAIL_DEVICE_TYPES];
    uint8_t ui8Status;
    uint8_t ui8Tmp;

    //
    // Button state at system reset will determine how we restore or clear
    // the RNP state and pairing information.
    //
    switch(g_ui8Buttons & ALL_BUTTONS)
    {
        //
        // Right button is pressed at startup.
        //
        case RIGHT_BUTTON:
        {
            //
            // Set ui8 to completely clear all configuration and state data
            // from the RNP.
            //
            pui8Value[0] = eRTI_CLEAR_CONFIG_CLEAR_STATE;
            break;
        }

        //
        // Left button is pressed at startup
        //
        case LEFT_BUTTON:
        {
            //
            // Clear just the state information. Configuration is kept.
            //
            pui8Value[0] = eRTI_CLEAR_STATE;
            break;
        }

        //
        // some other button or lack of button press operation was present at
        // startup.
        //
        default:
        {
            //
            // Restore all state and configuration
            //
            pui8Value[0] = eRTI_RESTORE_STATE;
            break;
        }
    }

    //
    // Execute the desired startup control setting. Clears or restore based on
    // button status.
    //
    ui8Status = RTI_WriteItem(RTI_CP_ITEM_STARTUP_CTRL, 1, pui8Value);
    if(ui8Status != RTI_SUCCESS)
    {
    }

    //
    // If we successfully read the startup control value and it is not set to
    // restore the previous state then we need to re-configure the RNP.
    // If we are set to RESTORE then we can skip this configuration section.
    //
    if((ui8Status == RTI_SUCCESS)  && (pui8Value[0] != eRTI_RESTORE_STATE))
    {
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_NODE_SUPPORTED_TGT_TYPES,
                                 RTI_MAX_NUM_SUPPORTED_TGT_TYPES,
                                 pucTargetList);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Application capabilities is a bit field that the application must
        // configure. It defines number of devices and profiles that this
        // node will be presenting to the network.
        //
        ui8Tmp = 0x22;
        ui8Status = RTI_WriteItem( RTI_CP_ITEM_APPL_CAPABILITIES, 1, &ui8Tmp);
        if (ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Write the list of supported device types to the RNP
        //
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_APPL_DEV_TYPE_LIST,
                                 RTI_MAX_NUM_DEV_TYPES, pui8DevList);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Write the list of supported profiles to the RNP.
        //
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_APPL_PROFILE_ID_LIST,
                                 RTI_MAX_NUM_PROFILE_IDS, pui8ProfileList);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Node capabilities is a bit field that controls security, power
        // and other node characteristics.
        //
        ui8Tmp = 0x07;
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_NODE_CAPABILITIES, 1, &ui8Tmp);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Write the Vendor ID number to the RNP.
        //
        pui8Value[0] = (uint8_t) (RTI_VENDOR_TEXAS_INSTRUMENTS & 0xFF);
        pui8Value[1] = (uint8_t) (RTI_VENDOR_TEXAS_INSTRUMENTS >> 8);
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_VENDOR_ID, 2, pui8Value);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Write the string version of vendor name to the RNP.
        //
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_VENDOR_NAME,
                                 sizeof(pui8VendorName), pui8VendorName);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Write the desired standby duty cycle to the RNP.
        //
        pui8Value[0] = (uint8_t) (1000 & 0xFF);
        pui8Value[1] = (uint8_t) (1000 >> 8);
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_STDBY_DEFAULT_DUTY_CYCLE, 2,
                                 pui8Value);
        if(ui8Status != RTI_SUCCESS)
        {
        }
    }
}

//*****************************************************************************
//
// Assemble an send a mouse report over the RF link.
//
//*****************************************************************************
void
ZIDSendMouseReport(uint8_t ui8MouseButtons, int8_t i8DeltaX,
                   int8_t i8DeltaY)
{
    uint8_t ui8TXOptions;
    tZIDReportDataCommand sZIDReport;

    //
    // Set reliable control pipe mode for mouse clicks.
    //
    if(ui8MouseButtons == 0)
    {
        ui8TXOptions = ZID_TX_OPTIONS_INTERRUPT_PIPE;
    }
    else
    {
        ui8TXOptions = ZID_TX_OPTIONS_CONTROL_PIPE;
    }

    //
    // Populate the ZIDReport with mouse constants.
    //
    sZIDReport.ui8Command = ZID_CMD_REPORT_DATA;
    sZIDReport.sReportRecord.ui8Length = ZID_MOUSE_REPORT_LENGTH;
    sZIDReport.sReportRecord.ui8Type = ZID_REPORT_TYPE_IN;
    sZIDReport.sReportRecord.ui8ID = ZID_STD_REPORT_MOUSE;
    sZIDReport.sReportRecord.uData.sMouseData.ui8Buttons = ui8MouseButtons;
    sZIDReport.sReportRecord.uData.sMouseData.i8DeltaX = i8DeltaX;
    sZIDReport.sReportRecord.uData.sMouseData.i8DeltaY = i8DeltaY;

    //
    // Set the link state to data pending and start the transmission.
    //
    g_vui8LinkState = LINK_STATE_NDATA;
    RTI_SendDataReq(g_ui8LinkDestIndex, RTI_PROFILE_ZID,
                    RTI_VENDOR_TEXAS_INSTRUMENTS, ui8TXOptions,
                    ZID_MOUSE_REPORT_LENGTH + 2, (uint8_t *) &sZIDReport);
}

//*****************************************************************************
//
// Assemble and send a keyboard data report.
//
//*****************************************************************************
void
ZIDSendKeyboardReport(tZIDKeyboardData *psKeys)
{
    uint8_t ui8TXOptions;
    tZIDReportDataCommand sZIDReport;

    //
    // Always use reliable control pipe mode for key presses.
    //
    ui8TXOptions = ZID_TX_OPTIONS_CONTROL_PIPE;

    //
    // Populate the ZID Report structure.
    //
    sZIDReport.ui8Command = ZID_CMD_REPORT_DATA;
    sZIDReport.sReportRecord.ui8Length = ZID_KEYBOARD_REPORT_LENGTH;
    sZIDReport.sReportRecord.ui8Type = ZID_REPORT_TYPE_IN;
    sZIDReport.sReportRecord.ui8ID = ZID_STD_REPORT_KEYBOARD;
    sZIDReport.sReportRecord.uData.sKeyboardData = *psKeys;

    //
    // Set the link state to data pending and start transmission.
    //
    g_vui8LinkState = LINK_STATE_NDATA;
    RTI_SendDataReq(g_ui8LinkDestIndex, RTI_PROFILE_ZID,
                   RTI_VENDOR_TEXAS_INSTRUMENTS, ui8TXOptions,
                   ZID_KEYBOARD_REPORT_LENGTH + 2,
                   (uint8_t *) &sZIDReport);
}

//*****************************************************************************
//
// This routine is for the case where the RC boots, and there already are
// pairing table entries, so which target does the RC pair with? This routine
// finds the last pairing entry that is valid.
//
// Note that this logic of selection is not very useful when the pairing table
// size is huge.
//
//*****************************************************************************
static unsigned char
ZIDSelectInitialTargetDevice(void)
{
    uint8_t ui8ReadStatus;
    uint8_t ui8ActiveEntries;

    //
    // Initialize the pairing index to invalid and assume by default we have
    // no active pairing table entries.
    //
    g_ui8LinkDestIndex = RTI_INVALID_PAIRING_REF;
    ui8ActiveEntries = 0;

    //
    // Read the number of active pairing entries.
    //
    ui8ReadStatus = RTI_ReadItemEx(RTI_PROFILE_RTI,
                                  RTI_SA_ITEM_PT_NUMBER_OF_ACTIVE_ENTRIES, 1,
                                  &ui8ActiveEntries);

    //
    // Determine if read was successful and their are active pairing entries
    // present.
    //
    if((ui8ReadStatus == RTI_SUCCESS) && (ui8ActiveEntries != 0))
    {
        //
        // Set the Current LinkDestIndex to the most recent active entry.
        //
        g_ui8LinkDestIndex = ui8ActiveEntries - 1;
        g_vui8LinkState = LINK_STATE_READY;
        RTI_WriteItemEx(RTI_PROFILE_RTI, RTI_SA_ITEM_PT_CURRENT_ENTRY_INDEX, 1,
                            &g_ui8LinkDestIndex);
    }
    return(g_ui8LinkDestIndex);
}

//*****************************************************************************
//
// Get a ZID report.  This is used when the other side of network sends us
// a report.  Currently we don't use this for anything.  It could be used for
// maintaining light status for things like CAPS LOCK across several keyboards.
//
//*****************************************************************************
static void
ZIDGetReport(uint8_t ui8SrcIndex, uint8_t *pui8Data)
{
    //
    // Do Nothing.
    //
    (void) ui8SrcIndex;
    (void) pui8Data;
}

//*****************************************************************************
//
// RTI Confirm function. Confirms the receipt of an Init Request (RTI_InitReq).
//
// Called by RTI_AsynchMsgProcess which we have placed in the main application
// context.
//
//*****************************************************************************
void
RTI_InitCnf(uint8_t ui8Status)
{
    uint8_t ui8MaxEntries, ui8DestIndex, ui8Value;

    //
    // Verify return status.
    //
    if (ui8Status == RTI_SUCCESS)
    {
        //
        // Make sure startup control is now set back to RESTORE mode.
        //
        ui8Value = eRTI_RESTORE_STATE;
        ui8Status = RTI_WriteItem(RTI_CP_ITEM_STARTUP_CTRL, 1, &ui8Value);

        //
        // Determine the maximum number of pairing table entries.
        //
        ui8Status = RTI_ReadItemEx(RTI_PROFILE_RTI,
                                   RTI_CONST_ITEM_MAX_PAIRING_TABLE_ENTRIES,
                                   1, &ui8MaxEntries);
        if(ui8Status != RTI_SUCCESS)
        {
        }

        //
        // Verify that read was successful.
        //
        if(ui8Status == RTI_SUCCESS)
        {
            //
            // Select initial target device type currently, this routine finds
            // the first pairing entry index, if one exists. If a valid entry
            // is found this function will set the global LinkDestIndex,
            // link state and send an RTI command to write back the current
            // active pairing entry.
            //
            ui8DestIndex = ZIDSelectInitialTargetDevice();
            if(ui8DestIndex == RTI_INVALID_PAIRING_REF)
            {
                //
                // A valid pairing index was not found.
                // Delay briefly so we don't overrun the RPI buffers.
                //
                ROM_SysCtlDelay(ROM_SysCtlClockGet() / (100*3));

                //
                // Send a Disable Sleep Request.  This will wake up the RPI
                // and then start a pairing sequence.
                //
                RTI_DisableSleepReq();

                //
                // Set the link state to pairing in process
                //
                g_vui8LinkState = LINK_STATE_PAIR;
                g_ui32RGBLPRFBlinkCounter = g_ui32SysTickCount;
            }
            else
            {
                //
                // Turn on the LED to show we paired and ready.
                //
                g_pui32RGBColors[GREEN] = 0x4000;
                RGBColorSet(g_pui32RGBColors);
            }
        }
    }
}

//*****************************************************************************
//
// RTI confirm function. Called by RTI_AsynchMsgProcess when a pairing
// request is confirmed.  Contains the status of the pairing attempt and
// if successful the destination index.
//
//*****************************************************************************
void
RTI_PairCnf(uint8_t ui8Status, uint8_t ui8DestIndex,
            uint8_t ui8DevType)
{
    (void) ui8DevType;

    //
    // Determine if the Pair attempt was successful.
    //
    if(ui8Status == RTI_SUCCESS)
    {
        //
        // Save the destination index we got back from our successful pairing.
        // Write that value back to the RNP so it knows this is the current
        // link to be used.
        //
        g_ui8LinkDestIndex = ui8DestIndex;
        ui8Status = RTI_WriteItemEx(RTI_PROFILE_RTI,
                                    RTI_SA_ITEM_PT_CURRENT_ENTRY_INDEX, 1,
                                    &g_ui8LinkDestIndex);

        //
        // Turn on the LED to show we paired and ready.
        //
        g_pui32RGBColors[GREEN] = 0x4000;
        RGBColorSet(g_pui32RGBColors);
    }
    else
    {
        //
        // Turn off the LED to show pairing failed.
        //
        g_pui32RGBColors[GREEN] = 0;
        RGBColorSet(g_pui32RGBColors);
    }
    g_vui8LinkState = LINK_STATE_READY;
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess if pairing was aborted.
// Currently not expected to get this call so just set the link state back to
// ready.
//
//*****************************************************************************
void
RTI_PairAbortCnf(uint8_t ui8Status)
{
    (void) ui8Status;

    //
    // Reset the link state.
    //
    if (LINK_STATE_PAIR == g_vui8LinkState)
    {
        g_vui8LinkState = LINK_STATE_READY;
    }
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when an allow pair
// request is recieved by the application processor.  This is not expected for
// this application.
//
//*****************************************************************************
void
RTI_AllowPairCnf(uint8_t ui8Status, uint8_t ui8DestIndex,
                 uint8_t ui8DevType)
{
    //
    // Do nothing. Controller does not trigger AllowPairReq() and hence is not
    // expecting this callback.
    //
    (void) ui8Status;
    (void) ui8DestIndex;
    (void) ui8DevType;
}

//*****************************************************************************
//
// RTI Confirm function.  Called by RTI_AsyncMsgProcess when a unpair request
// is confirmed by the RNP. Currently not implemented.
//
//*****************************************************************************
void
RTI_UnpairCnf(uint8_t ui8Status, uint8_t ui8DestIndex)
{
    //
    // unused arguments
    //
    (void) ui8Status;
    (void) ui8DestIndex;
}

//*****************************************************************************
//
// RTI indication function. Called by RTI_AsynchMsgProcess when the far side of
// the link is requesting to unpair. Currently not implemented.
//
//
//*****************************************************************************
void
RTI_UnpairInd(uint8_t ui8DestIndex)
{
    //
    // unused arguments
    //
    (void) ui8DestIndex;
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when a data send is
// confirmed. It is now clear to queue the next data packet.
//
//*****************************************************************************
void
RTI_SendDataCnf(uint8_t ui8Status)
{
    //
    // Set the link state to ready.
    //
    if (g_vui8LinkState == LINK_STATE_NDATA)
    {
        g_vui8LinkState = LINK_STATE_READY;
    }

    //
    // Turn on the LED to show we are back to ready state.
    //
    g_pui32RGBColors[GREEN] = 0x4000;
    RGBColorSet(g_pui32RGBColors);
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when the RxEnable
// request has been processed by the RNP. Not implemented.
//
//*****************************************************************************
void
RTI_RxEnableCnf(uint8_t ui8Status)
{
    //
    // Do nothing
    //
    (void) ui8Status;
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when the enable sleep
// request has been proccessed by the RNP.
//
//*****************************************************************************
void
RTI_EnableSleepCnf(uint8_t ui8Status)
{
    //
    // Do nothing
    //
    (void) ui8Status;
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when disable sleep
// request has been processed by the RNP. This is used during the init sequence
// as a trigger to start a pairing sequence if needed.
//
//*****************************************************************************
void
RTI_DisableSleepCnf(uint8_t ui8Status)
{
    (void) ui8Status;

    //
    // RNP is now awake, if we don't have a pairing link then start the pair
    // process.
    //
    if(g_ui8LinkDestIndex == RTI_INVALID_PAIRING_REF)
    {
        RTI_PairReq();
    }
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when the RNP would
// like to get the latest report data from the application processor.
//
//*****************************************************************************
void
RTI_ReceiveDataInd(uint8_t ui8SrcIndex, uint8_t ui8ProfileId,
                   uint16_t ui16VendorID, uint8_t ui8RXLinkQuality,
                   uint8_t ui8RXFlags, uint8_t ui8Length, uint8_t *pui8Data)
{
    uint8_t ui8Command;

    //
    // Verify the profile ID.
    //
    if (ui8ProfileId == RTI_PROFILE_ZID)
    {
        //
        // What is the command coming from the RNP.?
        //
        ui8Command = pui8Data[ZID_FRAME_CTL_IDX] & ZID_CMD_CODE_MASK;

        //
        // If this is a get report command then process it.
        //
        if (ui8Command == ZID_CMD_GET_REPORT)
        {
            //
            // parses the pui8Data into a mouse or keyboard report from the
            // RNP.
            //
            ZIDGetReport(ui8SrcIndex, pui8Data);
        }
    }
}

//*****************************************************************************
//
// RTI Confirm function. Called by RTI_AsynchMsgProcess when a standby request
// has been processed by the RNP.
//
//*****************************************************************************
void
RTI_StandbyCnf(uint8_t ui8Status)
{
    (void) ui8Status;
}

//*****************************************************************************
//
// RTI Callback function. The lower level UART and NPI layers have verified
// received an asynchronous message. Set the flag to indicate it needs
// processed.
//
//*****************************************************************************
void
RTI_AsynchMsgCallback(uint32_t ui32Data)
{
    (void) ui32Data;

    //
    // Set the flag to tell LPRF Main that we need to process a message.
    //
    HWREGBITW(&g_ui32Events, LPRF_EVENT) = 1;
}

//*****************************************************************************
//
//
//*****************************************************************************
void
RTI_AsynchMsgProcess(void)
{
    tRemoTIMsg sMsg;

    //
    // Get the msg from the UART low level driver
    //
    RemoTIUARTGetMsg((uint8_t *) &sMsg, NPI_MAX_DATA_LEN);

    if ((sMsg.ui8SubSystem & 0x1F) == RPC_SYS_RCAF)
    {
        switch((unsigned long) sMsg.ui8CommandID)
        {
            case RTIS_CMD_ID_RTI_INIT_CNF:
                RTI_InitCnf(sMsg.pui8Data[0]);
                break;

            case RTIS_CMD_ID_RTI_PAIR_CNF:
                RTI_PairCnf(sMsg.pui8Data[0], sMsg.pui8Data[1], sMsg.pui8Data[2]);
                break;

            case RTIS_CMD_ID_RTI_PAIR_ABORT_CNF:
                RTI_PairAbortCnf(sMsg.pui8Data[0]);
                break;

            case RTIS_CMD_ID_RTI_ALLOW_PAIR_CNF:
                RTI_AllowPairCnf(sMsg.pui8Data[0], sMsg.pui8Data[1],
                                 sMsg.pui8Data[2]);
                break;

            case RTIS_CMD_ID_RTI_SEND_DATA_CNF:
                RTI_SendDataCnf(sMsg.pui8Data[0]);
                break;

            case RTIS_CMD_ID_RTI_REC_DATA_IND:
                RTI_ReceiveDataInd(sMsg.pui8Data[0], sMsg.pui8Data[1],
                                   sMsg.pui8Data[2] | (sMsg.pui8Data[3] << 8),
                                   sMsg.pui8Data[4], sMsg.pui8Data[5],
                                   sMsg.pui8Data[6], &sMsg.pui8Data[7]);
                break;

            case RTIS_CMD_ID_RTI_STANDBY_CNF:
                RTI_StandbyCnf(sMsg.pui8Data[0] );
                break;

            case RTIS_CMD_ID_RTI_ENABLE_SLEEP_CNF:
                RTI_EnableSleepCnf(sMsg.pui8Data[0] );
                break;

            case RTIS_CMD_ID_RTI_DISABLE_SLEEP_CNF:
                RTI_DisableSleepCnf(sMsg.pui8Data[0] );
                break;

            case RTIS_CMD_ID_RTI_RX_ENABLE_CNF:
                RTI_RxEnableCnf(sMsg.pui8Data[0] );
                break;

            case RTIS_CMD_ID_RTI_UNPAIR_CNF:
                RTI_UnpairCnf(sMsg.pui8Data[0], sMsg.pui8Data[1]);
                break;

            case RTIS_CMD_ID_RTI_UNPAIR_IND:
                RTI_UnpairInd(sMsg.pui8Data[0]);
                break;

            default:
                // nothing can be done here!
                break;
        }
    }
}

//*****************************************************************************
//
//
//*****************************************************************************
void
LPRFInit(void)
{
    //
    // Enable the RemoTI UART pin muxing.
    // UART init is done in the RemoTI/uart_drv functions
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    ROM_GPIOPinConfigure(GPIO_PB0_U1RX);
    ROM_GPIOPinConfigure(GPIO_PB1_U1TX);
    ROM_GPIOPinTypeUART(GPIO_PORTB_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART1);

    //
    // Configure the CC2533 Reset pin
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOC);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTC_BASE, GPIO_PIN_7);

    RemoTIUARTInit(UART1_BASE);
    IntEnable(INT_UART1);

    NPI_Init(RTI_AsynchMsgCallback);
    RTI_Init();

    ZIDResetRNP();
    ZIDConfigParams();
    RTI_InitReq();
}

//*****************************************************************************
//
//
//*****************************************************************************
void
LPRFMain(void)
{
    int8_t i8DeltaX, i8DeltaY;
    uint8_t ui8Buttons;
    bool bSendIt, bButtonChange;

    //
    // First determine if we need to process a asynchronous message, such as
    // a send data confirmation or pairing confirmation.
    //
    while(HWREGBITW(&g_ui32Events, LPRF_EVENT) != 0)
    {
        //
        // Processes the message and calls the appropriate RTI callback.
        //
        RTI_AsynchMsgProcess();

        //
        // Clear the event flag.
        //
        if(RemoTIUARTGetRxMsgCount() == 0)
        {
            HWREGBITW(&g_ui32Events, LPRF_EVENT) = 0;
        }
    }

    if(g_vui8LinkState == LINK_STATE_PAIR)
    {
        //
        // Pairing in process so do a steady quick blink of the LED.
        //
        if(g_ui32SysTickCount > (g_ui32RGBLPRFBlinkCounter + 20))
        {
            //
            // 20 ticks have expired since we last toggled so turn off the
            // LED and reset the counter.
            //
            g_ui32RGBLPRFBlinkCounter = g_ui32SysTickCount;
            g_pui32RGBColors[GREEN] = 0;
            RGBColorSet(g_pui32RGBColors);
        }
        else if(g_ui32SysTickCount == (g_ui32RGBLPRFBlinkCounter + 10))
        {
            //
            // 10 ticks have expired since the last counter reset.  turn
            // on the RED LED.
            //
            g_pui32RGBColors[GREEN] = 0x4000;
            RGBColorSet(g_pui32RGBColors);
        }
    }
    else if((g_vui8LinkState == LINK_STATE_READY) &&
            (HWREGBITW(&g_ui32USBFlags, FLAG_CONNECTED) == 0))
    {
        //
        // Link is ready and USB is NOT connected so we can send mouse or
        // keyboard data. Get the desired mouse behaviors from the Motion
        // system.
        //
        bSendIt = MotionMouseGet(&i8DeltaX, &i8DeltaY, &ui8Buttons);
        if((g_ui8Buttons & ALL_BUTTONS) != (g_ui8LPRFButtonsPrev))
        {
            bButtonChange = true;
            g_ui8LPRFButtonsPrev = (g_ui8Buttons & ALL_BUTTONS);
        }
        else
        {
            bButtonChange = false;
        }

        if(bSendIt || bButtonChange)
        {
            //
            // Override the motion button action locally.  Motion currently
            // does not implement button behavior.
            //
            ui8Buttons = (g_ui8Buttons & LEFT_BUTTON) >> 3;
            ui8Buttons |= (g_ui8Buttons & RIGHT_BUTTON) << 0;

            //
            // Turn off the LED to show we are sending data
            //
            g_pui32RGBColors[GREEN] = 0;
            RGBColorSet(g_pui32RGBColors);

            //
            // Send the mouse report over the radio.
            //
            ZIDSendMouseReport(ui8Buttons, i8DeltaX, i8DeltaY);
        }

        //
        // Check if i had previously pressed some keyboard keys.  If not look
        // for new keys. If keyboard is pressed send the key release.
        //
        if(!g_bPressed)
        {
            //
            // Keyboard not already pressed. Get the Keyboard behaviors from
            // the motion system.
            //
            bSendIt = MotionKeyboardGet(&g_sZIDKeys.ui8Modifiers,
                                        &g_sZIDKeys.pui8Keys[0],
                                        &g_bModifierHold, &g_bKeyHold);

            //
            // If motion has something new then send it over the air and set
            // the flag to indicate a key is pressed.
            //
            if(bSendIt)
            {
                //
                // Turn off the LED to show we are sending data
                //
                g_pui32RGBColors[GREEN] = 0;
                RGBColorSet(g_pui32RGBColors);

                //
                // Send the keyboard report.
                //
                ZIDSendKeyboardReport(&g_sZIDKeys);
                g_bPressed = true;
            }
        }
        else
        {
            //
            // Clear the pressed flag to show we have complete this key press
            // and release cycle.
            //
            g_bPressed = false;

            //
            // A modifier key may have been pressed.  If hold is set then leave
            // the key in the report.  If hold is false clear the key.
            //
            if(!g_bModifierHold)
            {
                g_sZIDKeys.ui8Modifiers = 0;
                bSendIt = true;
            }

            //
            // A usage key may have been pressed.  If hold is set then leave
            // the key in the report.  If hold is false clear the key.
            //
            if(!g_bKeyHold)
            {
                g_sZIDKeys.pui8Keys[0] = 0;
                bSendIt = true;
            }

            //
            // If the report changed then send it.
            //
            if(bSendIt)
            {
                //
                // Turn off the LED to show we are ready to sending data
                //
                g_pui32RGBColors[GREEN] = 0;
                RGBColorSet(g_pui32RGBColors);

                //
                // Send the report
                //
                ZIDSendKeyboardReport(&g_sZIDKeys);
            }
        }
    }
}
