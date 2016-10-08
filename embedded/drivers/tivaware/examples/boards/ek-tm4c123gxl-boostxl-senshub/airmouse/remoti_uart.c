//*****************************************************************************
//
// remoti_uart.c - UART abstraction layer for RemoTI stack.
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

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_uart.h"
#include "driverlib/uart.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "utils/ringbuf.h"
#include "remoti_uart.h"

//*****************************************************************************
//
// Callback pointers used to send indications up to the NPI layer.
//
//*****************************************************************************
tRemoTICallback *g_pfnErrCallback;
tRemoTICallback *g_pfnTxCallback;
tRemoTICallback *g_pfnRxCallback;

//*****************************************************************************
//
// Number of messages in the RX UART queue awaiting processing.
//
//*****************************************************************************
uint_fast16_t g_ui16RxMsgCount;

//*****************************************************************************
//
// Ringbuffer objects for the Transmit and Receive channels.
//
//*****************************************************************************
tRingBufObject g_rbRemoTIRxRingBuf;
tRingBufObject g_rbRemoTITxRingBuf;

//*****************************************************************************
//
// Buffer Storage for Transmit and Receive.
//
//*****************************************************************************
uint8_t g_pui8RxBuf[REMOTI_UART_RX_BUF_SIZE];
uint8_t g_pui8TxBuf[REMOTI_UART_TX_BUF_SIZE];

//*****************************************************************************
//
// UART Transmit busy flag.  Used to determine when to prime transmitter with
// first byte of a new message.
//
//*****************************************************************************
bool g_bTxBusy;

//*****************************************************************************
//
// Base address of the UART peripheral used by this module.
//
//*****************************************************************************
uint32_t g_ui32UARTBase;

//*****************************************************************************
//
// Initializes the RemoTI UART driver interface to a remote network processor.
//
// \param ui32Base is the base address of the UART peripheral to be used.
// Caller must call SysCtlPeripheralEnable for this UART peripheral prior to
// calling this init funciton.
//
// This function will initialize the ring buffers used for transmit and receive
// of data to and from the RNP.  It also configures the UART peripheral for a
// default setting.  Enables receive interrupts.  Transmit interrupts are
// enabled when a transmit is in progress.  Master interrupt enable must be
// turned on by the application.
//
// \note Users of this driver are also responsible to assign
// RemoTIUARTIntHandler() as the interrupt routine associated with the UART
// peripheral of choice.
//
// \return None.
//
//*****************************************************************************
void
RemoTIUARTInit(uint32_t ui32Base)
{
    //
    // Save the UART peripheral base address for later use.
    //
    g_ui32UARTBase = ui32Base;

    //
    // Initialize the TX and RX ring buffers for storage of our data.
    //
    RingBufInit(&g_rbRemoTIRxRingBuf, g_pui8RxBuf, REMOTI_UART_TX_BUF_SIZE);
    RingBufInit(&g_rbRemoTITxRingBuf, g_pui8TxBuf, REMOTI_UART_TX_BUF_SIZE);

    //
    // Configure UART clock settings.
    //
    UARTConfigSetExpClk(ui32Base, SysCtlClockGet(), 115200,
                        (UART_CONFIG_WLEN_8 | UART_CONFIG_PAR_NONE |
                         UART_CONFIG_STOP_ONE | UART_FLOWCONTROL_NONE));

    //
    // Configure the UART FIFO. Enable the UART and Enable RX interrupts.
    //
    UARTFIFOLevelSet(ui32Base, UART_FIFO_TX1_8, UART_FIFO_RX1_8);
    UARTEnable(ui32Base);
    UARTFIFODisable(ui32Base);
    UARTIntEnable(ui32Base, UART_INT_RX);
}

//*****************************************************************************
//
// Puts a UART message on the line to the RNP.
//
// \param pui8Msg pointer to the data that will be put on the bus. Must be
// already formated with SOF, length and checksum by the caller.
//
// \param ui16Length the number of bytes that are to be put out the UART.
//
// This function copies the message to the ring buffer and then starts a
// transmission process. Application must assure that transmitter is not busy.
//
//*****************************************************************************
void
RemoTIUARTPutMsg(uint8_t* pui8Msg, uint_fast16_t ui16Length)
{
    bool bIntState;

    //
    // Capture the current state of the master interrupt enable.
    //
    bIntState = IntMasterDisable();

    //
    // Store the message in the ringbuffer for transmission.
    //
    RingBufWrite(&g_rbRemoTITxRingBuf, pui8Msg, ui16Length);

    //
    // If the UART transmit is idle prime the transmitter with first byte and
    // enable transmit interrupts.
    //
    if(!g_bTxBusy)
    {
        //
        // Enable the TX interrupts and start the transmission of the first
        // byte.
        //
        UARTIntEnable(g_ui32UARTBase, UART_INT_TX);
        UARTCharPutNonBlocking(g_ui32UARTBase,
                               RingBufReadOne(&g_rbRemoTITxRingBuf));

        //
        // Set the Transmit busy flag.
        //
        g_bTxBusy = true;
    }

    //
    // Restore the master interrupt enable to its previous state.
    //
    if(!bIntState)
    {
        IntMasterEnable();
    }

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Gets a message from the UART buffer.
//
// \param pui8Msg is a pointer to storage allocated by the caller where the
// message will be copied to.
//
// \param ui16Length is the length of the pui8Msg buffer.
//
// Copies a message from the UART buffer to the \e pui8Msg caller supplied
// storage.  If the caller supplied storage length is less than the next
// UART message length then the UART message is dumped and no data is returned.
// Therefore it is critical to make sure that caller supplies sufficient length
// for the longest anticipated message from the RNP.  256 bytes is recommended.
//
// \return None.
//
//*****************************************************************************
void
RemoTIUARTGetMsg(uint8_t* pui8Msg, uint_fast16_t ui16Length)
{
    bool bIntState;
    uint8_t ui8MsgLength;
    uint8_t ui8SOF;

    //
    // State previous state of master interrupt enable then disable all
    // interrupts.
    //
    bIntState = IntMasterDisable();

    //
    // Determine if a message is in the buffer available for the caller.
    //
    if(g_ui16RxMsgCount != 0)
    {
       //
       // Read out the SOF and Msg Length characters.
       //
       ui8SOF = RingBufReadOne(&g_rbRemoTIRxRingBuf);
       ui8MsgLength = RingBufReadOne(&g_rbRemoTIRxRingBuf);

       //
       // Make sure that the user buffer has room for the message and the
       // packet overhead bytes.
       //
       if((ui8MsgLength + 5) <= ui16Length)
       {
           //
           // We have enough room, so store the two already bytes in the user
           // buffer.
           //
           pui8Msg[0] = ui8SOF;
           pui8Msg[1] = ui8MsgLength;

           //
           // Read the remaining bytes to the user buffer.
           //
           RingBufRead(&g_rbRemoTIRxRingBuf, pui8Msg + 2, ui8MsgLength + 3);

       }
       else
       {
           //
           // The user did not provide enough room and we cannot easily put
           // the first couple of bytes back into the buffer.  Therefore,
           // we dump the remainder of the message.
           //
           RingBufAdvanceRead(&g_rbRemoTIRxRingBuf, ui8MsgLength + 3);
       }

       //
       // Decrement the msg counter. Now one less message in the UART buffer.
       //
       g_ui16RxMsgCount -= 1;
    }

    //
    // Restore the master interrupt enable state.
    //
    if(!bIntState)
    {
        IntMasterEnable();
    }

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Gets the current number of UART messages waiting to be processed.
//
// This function returns the number of unprocessed messages that are currently
// in the UART ring buffer.
//
// \return Current number of UART messages in the buffer.
//
//*****************************************************************************
uint_fast16_t
RemoTIUARTGetRxMsgCount(void)
{
    bool bIntState;
    uint_fast16_t ui16Temp;

    //
    // Turn off interrupts and save Interrupt enable state.
    //
    bIntState = IntMasterDisable();

    //
    // Copy message count to local variable.
    //
    ui16Temp = g_ui16RxMsgCount;

    //
    // Restore interrupt enable state.
    //
    if(!bIntState)
    {
        IntMasterEnable();
    }

    //
    // Return a snapshot of the message count.
    //
    return(ui16Temp);
}
//*****************************************************************************
//
// Registers are message receive callback.
//
// \param pfnCallback is a pointer to a tRemoTICallback function. This
// callback will be called when a frame is successfully completely received.
//
// Registers the \e pfnCallback as the function to be called when a UART
// message is complete and successful.  Callback is called from UART interrupt
// context. To unregister pass \b NULL as a parameter. There is only one active
// callback, subsequent calls to this function will override the earlier ones.
//
// \return None.
//
//*****************************************************************************
void
RemoTIUARTRegisterMsgRxCallback(tRemoTICallback *pfnCallback)
{
    bool bIntState;

    //
    // Turn off interrupts and save previous interrupt enable state.
    //
    bIntState = IntMasterDisable();

    //
    // Register the callback.
    //
    g_pfnRxCallback = pfnCallback;

    //
    // Restore interrupt master enable state.
    //
    if(!bIntState)
    {
        IntMasterEnable();
    }

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Register a callback for UART errors.
//
// \param pfnCallback is a pointer to a tRemoTICallback function.
//
// Register the \e pfnCallback that will be called if a UART error occurs such
// as frame check or unexpected start of frame. Only one callback is active at
// a time.  Calling this function more than once will override earlier
// callbacks.  Pass \b NULL to unregister.
//
// \return None.
//
//*****************************************************************************
void
RemoTIUARTRegisterErrCallback(tRemoTICallback *pfnCallback)
{
    bool bIntState;

    //
    // Turn off interrupts and save previous interrupt enable state.
    //
    bIntState = IntMasterDisable();

    //
    // Register the callback.
    //
    g_pfnErrCallback = pfnCallback;

    //
    // Restore interrupt master enable state.
    //
    if(!bIntState)
    {
        IntMasterEnable();
    }

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Register a callback for UART transmission complete..
//
// \param pfnCallback is a pointer to a tRemoTICallback function.
//
// Register the \e pfnCallback that will be called when a UART transmission is
// complete.  The Transmit buffer has just been emptied of its last byte. Only
// one callback is active at a time.  Calling this function more than once
// will override earlier callbacks.  Pass \b NULL to unregister.
//
// \return None.
//
//*****************************************************************************
void
RemoTIUARTRegisterTxCompleteCallback(tRemoTICallback *pfnCallback)
{
    bool bIntState;
    //
    // Turn off interrupts and save previous interrupt enable state.
    //
    bIntState = IntMasterDisable();

    //
    // Register the callback.
    //
    g_pfnTxCallback = pfnCallback;

    //
    // Restore interrupt master enable state.
    //
    if(!bIntState)
    {
        IntMasterEnable();
    }

    //
    // Finished.
    //
}

//*****************************************************************************
//
// Private function to manage UART received bytes.
//
// This function maintains the current state of the UART packet reception. It
// tracks start of frame, length expected, length received and the frame check
// sequence. Each new byte is captured and stored into the buffer. Status is
// updated and when a message is fully received and in the buffer the callback
// is called if present.
//
// \return None.
//
//*****************************************************************************
static void
RemoTIUARTRxHandler(void)
{
    static uint_fast8_t ui8SOFFlag;
    static uint_fast16_t ui16Length;
    static uint_fast16_t ui16Counter;
    static uint8_t ui8FrameCheck;
    static uint8_t ui8Command0;
    uint8_t ui8RxByte;

    //
    // Get the character from the hardware uart.
    //
    ui8RxByte = UARTCharGetNonBlocking(g_ui32UARTBase);

    //
    // Check if this is a start of frame character.
    //
    if(ui8RxByte == RPC_UART_SOF)
    {
        //
        // Check for error condition that all of prev msg was not captured.
        // send callback if present.
        //
        if(ui16Length != ui16Counter)
        {
            if(g_pfnErrCallback)
            {
                g_pfnErrCallback(REMOTI_UART_UNEXPECTED_SOF_ERR);
            }
        }
        else
        {
            //
            // This is a start of frame so set the flag and clear all the other
            // state indicators.
            //
            ui8SOFFlag = 1;
            ui16Length = 0;
            ui16Counter = 0;
            ui8FrameCheck = 0;
            ui8Command0 = 0;
        }
    }
    else
    {
        //
        // This is not a SOF char. if it is the char immediate after a SOF then
        // it is a length char and needs special handling.
        //
        if(ui8SOFFlag == 1)
        {
            //
            // Record the length in terms of total bytes in the UART frame.
            // RemoTI just sends number of bytes in the data payload we want
            // number of general bytes + SOF byte + Length byte + two command
            // bytes + frame check byte.
            //
            ui16Length = ui8RxByte + 5;

            //
            // Clear the SOF flag.
            //
            ui8SOFFlag = 0;
        }
    }

    //
    // load the new byte in to the ring buffer for later use. unless we are
    // receiving past the end of an expected message's length.
    //
    if(ui16Counter <= ui16Length)
    {
        RingBufWriteOne(&g_rbRemoTIRxRingBuf, ui8RxByte);
        //
        // increment the counter to track how many bytes are in this msg.
        //
        ui16Counter++;

        if(ui16Counter == 3)
        {
            ui8Command0 = ui8RxByte;
        }
    }
    else if(g_pfnErrCallback)
    {
        //
        // Alert to the user code that RX Msg Length was greater than expected
        //
        g_pfnErrCallback(REMOTI_UART_RX_LENGTH_ERR);
    }

    //
    // Check if this is the end of the message and manage callbacks
    //
    if(ui16Length == ui16Counter)
    {
        //
        // compare the current Frame Check to the received frame check
        // if not equal then call error callback if present.
        //
        if(ui8FrameCheck != ui8RxByte)
        {
            //
            // Advance read index which effectively dumps the erroneous msg.
            //
            RingBufAdvanceRead(&g_rbRemoTIRxRingBuf, ui16Counter);

            //
            // If a callback is registered, call it.
            //
            if(g_pfnErrCallback)
            {
                //
                // Alert the user that the Frame check sequence failed.
                //
                g_pfnErrCallback(REMOTI_UART_RX_FCS_ERR);
            }
        }
        else
        {
            //
            // Message was successfully received and copied to local buffers.
            // Frame check was valid.  Increment the message counter and call
            // the receive callback.
            //
            g_ui16RxMsgCount += 1;
            if(g_pfnRxCallback)
            {
                g_pfnRxCallback(ui8Command0);
            }
        }
    }
    else if(!ui8SOFFlag)
    {
        //
        // calculate the frame check as we go.
        //
        ui8FrameCheck ^= ui8RxByte;
    }
}

//*****************************************************************************
//
// UART interrupt handler.
//
// This is the interrupt handler for the UART interrupts from the UART
// peripheral that has been associated with the remote network processor.
//
// \return None.
//*****************************************************************************
void
RemoTIUARTIntHandler(void)
{
    uint8_t ui8TxByte;

    //
    // Process all available interrupts while we are in this routine.
    //
    do
    {
        //
        // Check if a receive interrupt is pending.
        //
        if(UARTIntStatus(g_ui32UARTBase, 1) & UART_INT_RX)
        {
            //
            // A char was received, process it.  Do this first so it does not
            // get overwritten by future bytes.
            //
            UARTIntClear(g_ui32UARTBase, UART_INT_RX);
            RemoTIUARTRxHandler();
        }

        //
        // Check if a transmit interrupt is pending.
        //
        if(UARTIntStatus(g_ui32UARTBase, 1) & UART_INT_TX)
        {
            //
            // A byte transmission completed so load another byte or turn off
            // tx interrupts.
            //
            if(RingBufUsed(&g_rbRemoTITxRingBuf))
            {
                //
                // We still have more stuff to transfer so read the next byte
                // from the buffer and load it into the UART.  Finally clear
                // the pending interrupt status.
                //
                UARTIntClear(g_ui32UARTBase, UART_INT_TX);
                ui8TxByte = RingBufReadOne(&g_rbRemoTITxRingBuf);
                UARTCharPutNonBlocking(g_ui32UARTBase, ui8TxByte);
            }
            else
            {
                //
                // Transmission is complete and the internal buffer is empty.
                // Therefore, disable TX interrupts until next transmit is
                // started by the user.
                //
                UARTIntClear(g_ui32UARTBase, UART_INT_TX);
                UARTIntDisable(g_ui32UARTBase, UART_INT_TX);

                //
                // Clear the transmitter busy flag.
                //
                g_bTxBusy = false;

                //
                // Callback to the TX Complete callback function.
                //
                if(g_pfnTxCallback)
                {
                    g_pfnTxCallback(0);
                }
            }
        }
    //
    // Continue to process the interrupts until there are no more pending.
    //
    }while(UARTIntStatus(g_ui32UARTBase, 1) & (UART_INT_RX | UART_INT_TX));

    //
    // Finished.
    //
}
