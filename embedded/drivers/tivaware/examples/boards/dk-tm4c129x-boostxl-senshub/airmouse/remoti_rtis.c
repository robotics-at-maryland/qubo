//*****************************************************************************
//
// remoti_rtis.c - RemoTI surrogate API.  Use with Remote Network Processor.
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
#include <stdbool.h>
#include "driverlib/sysctl.h"
#include "remoti_uart.h"
#include "remoti_npi.h"
#include "remoti_rti.h"
#include "remoti_rtis.h"

//*****************************************************************************
//
// Global variable that holds clock frequency.
//
//*****************************************************************************
extern uint32_t g_ui32SysClock;

//*****************************************************************************
//
// The current state of the RemoTI surrogate stack.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8RTISState;

//*****************************************************************************
//
// Get the current state of the RemoTI link.
//
// Returns the current state of the RTIS layer and link to the RemoTI network.
//
// \return Returns the current value of \e g_vui8RTISState. Will be one of the
// states \e RTIS_STATE_INIT, \e RTIS_STATE_READY or \e
// RTIS_STATE_NETWORK_LAYER_BRIDGE
//
//*****************************************************************************
uint_fast8_t
RTIS_GetLinkState(void)
{
    //
    // Return the state.
    //
    return(g_vui8RTISState);
}

//*****************************************************************************
//
// Initialize the RTIS layer.
//
// Simple init function that initialize the \e g_vui8RTISState to a known
// condition.
//
// \return None.
//
//*****************************************************************************
void
RTI_Init(void)
{
    g_vui8RTISState = RTIS_STATE_READY;
}

//*****************************************************************************
//
// Wrapper which the application can use to read parameters from the RNP.
//
// \param ui8ProfileId the application profile associated with the item to
// read.
// \param ui8ItemId the identifier value for the item to read.
// \param ui8Length the length in bytes to be read.
// \param pui8Value user provided storage where the item will be stored.
//
// This function allocates a tRemoTIMsg on the stack and populates with the
// fields provided. This message is then copied to the UART buffers and
// transmitted as via \e NPI_SendSynchData. If successful the value read is
// copied back into the \e pui8Value buffer provided by the caller. If not
// successful \e RTI_ERROR_NO_RESPONSE is returned.
//
// \return Returns \e RTI_SUCCESS or \e RTI_ERROR_NO_RESPONSE.
//
//*****************************************************************************
uint_fast8_t
RTI_ReadItemEx(uint8_t ui8ProfileID, uint8_t ui8ItemID, uint8_t ui8Length,
               uint8_t *pui8Value)
{
    tRemoTIMsg sMsg;
    uint32_t ui32Index;

    //
    // Initialize the remoTI message structure.  Depending on the version of the
    // remoTI stack being used a compile time switch determines which fields are
    // present and how they are populated. Version 1.2.1 did not include the
    // ui8ProfileId element.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
#if (defined REMOTI_1_2_1) && (REMOTI_1_2_1 == true)
    (void) ucProfileID;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_READ_ITEM;
    sMsg.ui8Length    = 2;
    sMsg.pui8Data[0]  = ucItemID;
    sMsg.pui8Data[1]  = ui8Length;
#else
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_READ_ITEM_EX;
    sMsg.ui8Length    = 3;
    sMsg.pui8Data[0]  = ui8ProfileID;
    sMsg.pui8Data[1]  = ui8ItemID;
    sMsg.pui8Data[2]  = ui8Length;
#endif

    //
    // Send the data. If successful copy the item read to the caller's buffer.
    if(NPI_SendSynchData(&sMsg))
    {
        //
        // Copy the read item bytes to the caller's buffer.
        //
        for(ui32Index = 0; ui32Index < (sMsg.ui8Length - 1)  ; ui32Index++)
        {
            pui8Value[ui32Index] = sMsg.pui8Data[ui32Index + 1];
        }

        //
        // Return with success indicated.
        //
        return(RTI_SUCCESS);
    }

    //
    // Return with the no response error condition.
    //
    return(RTI_ERROR_NO_RESPONSE);
}

//*****************************************************************************
//
// Make the RTI_WriteItem function extern in this compilation unit. This
// provides non-inline definitions for these functions to handle the cases
// where the compiler chooses not to inline the functions (which is a valid
// choice for the compiler to make).
//
//*****************************************************************************
extern uint_fast8_t RTI_ReadItem(uint8_t ui8ItemID, uint8_t ui8Length,
                                 uint8_t *pui8Value);

//*****************************************************************************
//
// Write a configuration parameter to the RNP.
//
// \param ui8ProfileId the application profile to be used for this transaciton.
// \param ui8ItemId the identifier of the item to be written.
// \param ui8Length the number of bytes to write.
// \param pui8Value a pointer to the bytes to be written.
//
// This function allocates a tRemoTIMsg on the stack and populates with the
// fields provided. This message is then copied to the UART buffers and
// transmitted via \e NPI_SendSynchData. The \e ui8ItemID is written with the
// bytes of data pointed to by \e pui8Data.
//
// \return Returns \e RTI_SUCCESS or \e RTI_ERROR_NO_RESPONSE.
//
//*****************************************************************************
uint_fast8_t
RTI_WriteItemEx(uint_fast8_t ui8ProfileID, uint_fast8_t ui8ItemID,
                uint_fast8_t ui8Length, const uint8_t *pui8Value)
{
    tRemoTIMsg sMsg;
    uint_fast8_t ui8Index;

    //
    // Initialize the tRemoTIMsg object header information.
    // Note that version 1.2.1 of the RemoTI stack does not use the profile ID.
    // A compile time switch is used to determine how the message should be
    // structured.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
#if (defined REMOTI_1_2_1) && (REMOTI_1_2_1 == true)
    (void) profileId;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_WRITE_ITEM;
    sMsg.ui8Length    = ui8Length + 2;
    sMsg.pui8Data[0]  = ucItemID;
    sMsg.pui8Data[1]  = ui8Length;
#else
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_WRITE_ITEM_EX;
    sMsg.ui8Length    = ui8Length + 3;
    sMsg.pui8Data[0]  = ui8ProfileID;
    sMsg.pui8Data[1]  = ui8ItemID;
    sMsg.pui8Data[2]  = ui8Length;
#endif

    //
    // Copy payload data.
    //
    for(ui8Index = 0; ui8Index < ui8Length; ui8Index++)
    {
        sMsg.pui8Data[ui8Index+3] = pui8Value[ui8Index];
    }

    //
    // Send the message and interpret the return message.
    //
    if(NPI_SendSynchData(&sMsg))
    {
        //
        // Return Success.
        //
        return(RTI_SUCCESS);
    }

    //
    // Return an error to indicate the RNP did not respond before the timeout.
    //
    return(RTI_ERROR_NO_RESPONSE);
}

//*****************************************************************************
//
// Make the RTI_WriteItem function extern in this compilation unit. This
// provides non-inline definitions for these functions to handle the cases
// where the compiler chooses not to inline the functions (which is a valid
// choice for the compiler to make).
//
//*****************************************************************************
extern uint_fast8_t RTI_WriteItem(uint8_t ui8ItemID, uint8_t ui8Length,
                                  const uint8_t *pui8Value);

//*****************************************************************************
//
// Send a initialize request to the RNP.
//
// Creates and sends a tRemoTIMsg which contains a request to initialize the
// the RNP using the parameters previously provided.
//
// \note This is an asynchronous request.
//
// \sa RTI_InitCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_InitReq( void )
{
    tRemoTIMsg sMsg;

    //
    // Load the tRemoTIMsg with appropriate parameters.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_INIT_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send init request to RNP RTIS asynchronously.
    //
    NPI_SendAsynchData( &sMsg );
}

//*****************************************************************************
//
// Sends pairing request to the RNP.
//
// Creates and sends a tRemoTIMsg to the RNP to initiate a pairing request.
// If the system is properly configured this will start a pairing process over
// the air.
//
// \note This is an asynchronous request.
//
// \sa RTI_PairCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_PairReq( void )
{
    tRemoTIMsg sMsg;

    //
    // Populate the pair request message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_PAIR_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send pair request to RNP RTIS asynchronously.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Request that the RNP abort an ongoing pair request.
//
// This function will build and send a tRemoTIMsg that will cancel the current
// pairing operation.
//
// \note This is an asynchronous request.
//
// \sa RTI_PairAbortCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_PairAbortReq( void )
{
    tRemoTIMsg sMsg;

    //
    // Populate the pair abort request message
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_PAIR_ABORT_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send pair abort request to RNP RTIS asynchronously
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Request that the RNP allow other to pair to this node.
//
// This function will build and send a tRemoTIMsg that will start an allow
// pairing session.
//
// \note This is an asynchronous request.
//
// \sa RTI_AllowPairCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_AllowPairReq(void)
{
    tRemoTIMsg sMsg;

    //
    // Populate the allow pair request message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_ALLOW_PAIR_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send the allow pair request message to the RNP RTIS asynchronously.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Request that the RNP abort an ongoing allow pair session
//
// This function will build and send a tRemoTIMsg that will cancel the current
// pairing operation.
//
// \note This is an asynchronous request.
//
// \sa RTI_AllowPairAbortCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_AllowPairAbortReq(void)
{
    tRemoTIMsg sMsg;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_ALLOW_PAIR_ABORT_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Request to un-pair from a particular destination index.
//
// \param ui8DestIndex the destination index that caller wishes to unpair from.
//
// This function will build and send a unpair request message to the RNP.
//
// \note This is an asynchronous request.
//
// \sa RTI_UnpairCnf()
//
// \returns None.
//
//*****************************************************************************
void
RTI_UnpairReq(uint8_t ui8DestIndex)
{
    tRemoTIMsg sMsg;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_UNPAIR_REQ;
    sMsg.ui8Length    = 1;
    sMsg.pui8Data[0]  = ui8DestIndex;

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Send a Data Packet to the RNP.
//
// \param ui8DestIndex the destination index of the pairing partner that will
// receive this data packet.
// \param ui8ProfileId the profile identifier used for this transaction.
// \param ui16VendorID the vendor identifier for this transaction.
// \param ui8TXOptions Transmit options such as \b RTI_TX_OPTION_BROADCAST or
// \b RTI_TX_OPTION_ACKNOWLEDGED. This is an OR'd combination of bit flags.
// \param ui8Length the length of the data to be sent.
// \param pui8Data a pointer to the data in caller provided storage.
//
// This function builds and sends a data request to the RNP.  This request
// builds the message format on top of the tRemoTIMsg base structure elements.
//
// \note This is an asynchronous request.
//
// \sa RTI_SendDataCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_SendDataReq(uint8_t ui8DestIndex, uint8_t ui8ProfileID,
                uint16_t ui16VendorID, uint8_t ui8TXOptions,
                uint8_t ui8Length, uint8_t *pui8Data)
{
    tRemoTIMsg sMsg;
    uint32_t ui32Index;
    uint8_t *pui8Tmp;

    //
    // Cast the vendor identifier to a byte pointer to allow endian swapping.
    //
    pui8Tmp = (uint8_t*) &ui16VendorID;

    //
    // Populate the message header bytes.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_SEND_DATA_REQ;
    sMsg.ui8Length    = ui8Length + 6;
    sMsg.pui8Data[0]  = ui8DestIndex;
    sMsg.pui8Data[1]  = ui8ProfileID;
    sMsg.pui8Data[2]  = pui8Tmp[1];
    sMsg.pui8Data[3]  = pui8Tmp[0];
    sMsg.pui8Data[4]  = ui8TXOptions;
    sMsg.pui8Data[5]  = ui8Length;

    //
    // Populate the payload section of the message.
    //
    for(ui32Index = 0; ui32Index < ui8Length; ui32Index++)
    {
        sMsg.pui8Data[ui32Index + 6] = pui8Data[ui32Index];
    }

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Sends a standby request to the RNP.
//
// Generate and send a tRemoTIMsg to request the RNP enter standby mode.
//
// \note This is an asynchronous message.
//
// \sa RTI_StandbyCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_StandbyReq(uint8_t ui8Mode)
{
    tRemoTIMsg sMsg;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_STANDBY_REQ;
    sMsg.ui8Length    = 1;
    sMsg.pui8Data[0]  = ui8Mode;

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Send a request to the RNP to enable receive.
//
// \param ui32Duration the length of time in milliseconds to enable the
// receiver.  Must be between 1 and 0xFFFE or \b RTI_RX_ENABLE_OFF or
// \b RTI_RX_ENABLE_ON.
//
// This function sends a request to the RNP to turn on the receiver for the
// specified duration of time.
//
// \note This is an asynchronous request.
//
// \sa RTI_RxEnableCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_RxEnableReq(uint32_t ui32Duration)
{
    tRemoTIMsg sMsg;
    uint8_t* pui8Tmp;

    //
    // Case the duration value to a byte pointer to enable endian swapping.
    //
    pui8Tmp = (uint8_t *) &ui32Duration;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_RX_ENABLE_REQ;
    sMsg.ui8Length    = 4;
    sMsg.pui8Data[0]  = pui8Tmp[3];
    sMsg.pui8Data[1]  = pui8Tmp[2];
    sMsg.pui8Data[2]  = pui8Tmp[1];
    sMsg.pui8Data[3]  = pui8Tmp[0];

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Send a message to the RNP to enable sleep mode.
//
// Builds and sends a tRemoTIMsg to the RNP that enables sleep mode.
//
// \note This is an asynchronous request.
//
// \sa RTI_EnableSleepCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_EnableSleepReq( void )
{
//
// Compile time switch is used to determine if the power saving modes are
// compiled into the library.  These settings must be the same on the
// application processor and the RNP.
//
#if (defined RNP_POWER_SAVING) && (RNP_POWER_SAVING == TRUE)
    tRemoTIMsg sMsg;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_ENABLE_SLEEP_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);

#endif
}

//*****************************************************************************
//
// Send a message to the RNP to disable sleep mode.
//
// This function will build and send a message to the RNP to disable sleep
// mode.
//
// \note This is an asynchronous request.
//
// \sa RTI_DisableSleepCnf()
//
// \return None.
//
//*****************************************************************************
void
RTI_DisableSleepReq( void )
{
    tRemoTIMsg sMsg;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_DISABLE_SLEEP_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send the Message.
    //
    NPI_SendAsynchData(&sMsg);
}

//*****************************************************************************
//
// Send a request to the RNP to perform a software reset.
//
// This function will build and send a message over the UART to the RNP to
// request that the RNP perform a software reset on itself. This function
// will generate a spin delay of 200 milliseconds to allow the RNP time to
// complete the reset operation.
//
// \note This is an asynchronous request without a corresponding confirmation.
//
// \return None.
//*****************************************************************************
void
RTI_SwResetReq(void)
{
    tRemoTIMsg sMsg;

    //
    // Populate the message.
    //
    sMsg.ui8SubSystem = RPC_SYS_RCAF;
    sMsg.ui8CommandID = RTIS_CMD_ID_RTI_SW_RESET_REQ;
    sMsg.ui8Length    = 0;

    //
    // Send the Software Reset Request
    //
    NPI_SendAsynchData(&sMsg);

    //
    // Stall for 200 milliseconds to give the RNP time to do its reset.
    //
    SysCtlDelay(g_ui32SysClock / (3 * 5));
}
