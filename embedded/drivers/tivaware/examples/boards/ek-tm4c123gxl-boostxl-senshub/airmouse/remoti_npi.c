//*****************************************************************************
//
// remoti_npi.c - Network Processor Interface for RemoTI Zigbee Stack.
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
#include "driverlib/sysctl.h"
#include "remoti_uart.h"
#include "remoti_npi.h"

//*****************************************************************************
//
// The callback that has been registered by the RTIS layer.  Called when a
// Asynchronous message is recieved from the UART.
//
//*****************************************************************************
tRemoTICallback *g_pfnRTIRxCallback;

//*****************************************************************************
//
// Flag to indicate if a response to a synchronous message request has been
// received.
//
//*****************************************************************************
volatile bool g_vbSyncResponse;

//*****************************************************************************
//
// An Error Log that records UART error codes. from the REMOTI UART driver.
//
//*****************************************************************************
volatile uint32_t pui32ErrLog[128];

//*****************************************************************************
//
// Callback function for message complete.
//
// \param ui32Command0 is the command zero field of the message from the RNP.
// This field is used to indicate if this is a asynchronous or synchronous
// message.
//
// Interprets the \e ui32Command0 parameter to either call the higher RTI layer
// callback function or set the flag to indicate that a synchronous message has
// been responded to. Synchronous messages are responded to immediately without
// any other messages in between them.  The NPI will do a spin wait, with
// timeout, on the \e g_vbSyncResponse flag to await responses for synchronous
// messages sent. For asynchronous messages the higher level RTI, ZID or
// application should handle the callbacks to interpret responses.
//
// \return None.
//
//*****************************************************************************
void
NPI_MsgRxCallback(uint32_t ui32Command0)
{
    //
    // Check the command type
    //
    if((ui32Command0 & RPC_CMD_TYPE_MASK) == RPC_CMD_AREQ)
    {
        //
        // An asynchronous message was received so call the upper RTI layer
        // callback function if there is one.
        //
        if(g_pfnRTIRxCallback)
        {
            g_pfnRTIRxCallback(0);
        }
    }
    else
    {
        //
        // A synchronous message was received set the flag to indicate this.
        //
        g_vbSyncResponse = true;
    }
}

//*****************************************************************************
//
// Callback function. Called in the event of a UART messag error.
//
// \param ui32Error is the type of error that occurred as defined by one of
// \b REMOTI_UART_RX_LENGTH_ERR, \b REMOTI_UART_RX_FCS_ERR or
// \b REMOTI_UART_UNEXPECTED_SOF_ERR.
//
// Currently this function merely logs error codes for later examination by
// the programmer.  The user is free to implement more sophisticated error
// handling as needed for their application.
//
// \return None.
//
//*****************************************************************************
void
NPI_ErrCallback(uint32_t ui32Error)
{
    static uint32_t ui32Index;

    //
    // Log the error type to the error buffer.
    //
    pui32ErrLog[ui32Index] = ui32Error;

    //
    // Increment the error index variable. and wrap it back to zero as needed.
    //
    ui32Index++;
    if(ui32Index >= 128)
    {
        ui32Index = 0;
    }

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Initialize the NPI layer of the stack.
//
// \param pfnCallback is a pointer to a tRemoTICallback function that is called
// by this layer back up to the RTI layer when an asynchronous message is
// received over the UART.
//
// Performs NPI network processor interface layer initialization.
//
// \return None.
//
//*****************************************************************************
void
NPI_Init(tRemoTICallback *pfnCallback)
{
    //
    // Store the callback function pointer.
    //
    g_pfnRTIRxCallback = pfnCallback;

    //
    // Register the error and receive callbacks with the UART layer to callback
    // up to this (NPI) layer.
    //
    RemoTIUARTRegisterMsgRxCallback(NPI_MsgRxCallback);
    RemoTIUARTRegisterErrCallback(NPI_ErrCallback);

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Wrap the appropriate header and footer information around a \e tRemotTIMsg.
//
// \param psMsg a pointer to the tRemoTIMsg object that is to be prepared
// for transmit on the UART. \e ui8SOF element will be modified. Final byte of
// message will become the frame check.
//
// This function sets the header bytes and calculates the check sum for a
// message that is about to go out on the UART to the RNP.
//
// \return None.
//
//*****************************************************************************
void
NPI_MsgWrap(tRemoTIMsg* psMsg)
{
    uint32_t ui32Index;

    //
    // Set the first message byte to the start of frame indicator
    //
    psMsg->ui8SOF = RPC_UART_SOF;

    //
    // Clear the frame check byte. Then begin frame check calculation with
    // the header bytes that come after the SOF.
    //
    psMsg->pui8Data[psMsg->ui8Length] = 0;
    psMsg->pui8Data[psMsg->ui8Length] ^= psMsg->ui8Length;
    psMsg->pui8Data[psMsg->ui8Length] ^= psMsg->ui8SubSystem;
    psMsg->pui8Data[psMsg->ui8Length] ^= psMsg->ui8CommandID;

    //
    // XOR each byte of the data payload of the message into the last byte of
    // the payload which is the frame check.
    //
    for(ui32Index = 0; ui32Index < psMsg->ui8Length; ui32Index++)
    {
        //
        // Sequentially XOR each payload byte into the last byte of the
        // message.
        //
        psMsg->pui8Data[psMsg->ui8Length] ^= psMsg->pui8Data[ui32Index];
    }

    //
    // Finished
    //
}
//*****************************************************************************
//
// Send an asynchronous message to the RNP.
//
// \param psMsg a pointer to the tRemoTIMsg object to be sent.
//
// This function sets the \e ui8SubSystem field flags to indicate this is an
// asynchronous message.  It then calls NPI_MsgWrap() to set frame check and
// start of frame.  Finally it puts the message into the outgoing UART buffer
// for transmission to the RNP.
//
// \return None.
//
//*****************************************************************************
void
NPI_SendAsynchData(tRemoTIMsg *psMsg)
{
    uint8_t* pui8Buf;

    //
    // cast the tRemoTIMsg structure to an array of bytes.
    //
    pui8Buf = (uint8_t *) psMsg;

    //
    // Add the proper RPC type to the header.
    //
    psMsg->ui8SubSystem &= RPC_SUBSYSTEM_MASK;
    psMsg->ui8SubSystem |= RPC_CMD_AREQ;

    //
    // Add the SOF and FCS bytes to the msg
    //
    NPI_MsgWrap(psMsg);

    //
    // Put the msg in the UART buffer and start transmission.
    //
    RemoTIUARTPutMsg(pui8Buf, psMsg->ui8Length + 5);

    //
    // Finished.
    //
}


//*****************************************************************************
//
// Send a synchronous data message and wait for the response with a timeout.
//
// \param psMsg the tRemoTIMsg to be sent. If successful and a reply was
// received then this will also hold the reply message when this function
// returns.
//
// This function sets the \e ui8SubSystem element to indicate this is a
// synchronous message transfer. It then calls NPI_MsgWrap() to set the start
// of frame and frame check bytes.  Finally  it puts the message in the UART
// buffer, starts transmission and waits for a response.  If a response is
// not given false is returned. If the response was received before the timeout
// then the response message is loaded into \e psMsg and true is returned.
//
// \return Returns false if response was not received before timeout. Returns
// true and copies response to \e psMsg if response was received.
//
//*****************************************************************************
bool
NPI_SendSynchData(tRemoTIMsg *psMsg)
{
    uint8_t *pui8Buf;
    uint32_t ui32Ticks;

    //
    // Cast the tRemoTIMsg to a byte array.
    //
    pui8Buf = (uint8_t *) psMsg;

    //
    // Add the proper RPC type to the header.
    //
    psMsg->ui8SubSystem &= RPC_SUBSYSTEM_MASK;
    psMsg->ui8SubSystem |= RPC_CMD_SREQ;

    //
    // Add the SOF and FCS bytes to the msg.
    //
    NPI_MsgWrap(psMsg);

    //
    // Clear the flag to show we do not have a response to this outgoing msg.
    // Clear the tick counter to start our timeout sequence.
    //
    g_vbSyncResponse = false;
    ui32Ticks = 0;

    //
    // Send the msg and wait for a response to come back
    //
    RemoTIUARTPutMsg(pui8Buf, psMsg->ui8Length + 5);
    do
    {
        //
        // Delay about 10 milliseconds waiting for the response.
        //
        SysCtlDelay(SysCtlClockGet() / (100 * 3));
        ui32Ticks++;

    //
    // Continue waiting for the response until it comes in or we hit the
    // timeout limit.
    //
    }while((!g_vbSyncResponse) && (ui32Ticks < NPI_SYNC_MSG_TIMEOUT));

    //
    //Return failure if we had a timeout
    //
    if((ui32Ticks >= NPI_SYNC_MSG_TIMEOUT) || (!g_vbSyncResponse))
    {
        return(false);
    }

    //
    // Go get the response from the UART and put it into the msg buffer
    //
    RemoTIUARTGetMsg(pui8Buf, psMsg->ui8Length + 5);

    //
    // Return Success if the response came back in time.
    //
    return(true);
}
