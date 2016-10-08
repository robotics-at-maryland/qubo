//*****************************************************************************
//
// packet_handler.c
//
// Copyright (c) 2006-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************

//*****************************************************************************
//
//! \defgroup packet_handler Packet Handler API
//! This section describes the functions that are responsible for handling the
//! serial packets in the format that is supported by the serial flash loader.
//! These functions call the UART transfer functions that are discussed in the
//! UART Handler API section.
//! @{
//
//*****************************************************************************
#include <stdbool.h>
#include <stdint.h>
#include "packet_handler.h"
#include "uart_handler.h"

//****************************************************************************
//
// local declarations
//
//****************************************************************************
uint8_t CheckSum(uint8_t *pui8Data, uint8_t ui8Size);

//****************************************************************************
//
//! AckPacket() sends an Acknowledge a packet.
//!
//! This function acknowledges a packet has been received from the device.
//!
//! \return The function returns zero to indicated success while any non-zero
//! value indicates a failure.
//
//****************************************************************************
int32_t
AckPacket(void)
{
    uint8_t ui8Ack;

    ui8Ack = COMMAND_ACK;
    return(UARTSendData(&ui8Ack, 1));
}

//****************************************************************************
//
//! NakPacket() sends a No Acknowledge packet.
//!
//! This function sends a no acknowledge for a packet that has been
//! received unsuccessfully from the device.
//!
//! \return The function returns zero to indicated success while any non-zero
//! value indicates a failure.
//
//****************************************************************************
int32_t
NakPacket(void)
{
    uint8_t ui8Nak;

    ui8Nak = COMMAND_NAK;
    return(UARTSendData(&ui8Nak, 1));
}

//*****************************************************************************
//
//! GetPacket() receives a data packet.
//!
//! \param pui8Data is the location to store the data received from the device.
//! \param pui8Size is the number of bytes returned in the pui8Data buffer that
//! was provided.
//!
//! This function receives a packet of data from UART port.
//!
//! \returns The function returns zero to indicated success while any non-zero
//! value indicates a failure.
//
//*****************************************************************************
int32_t
GetPacket(uint8_t *pui8Data, uint8_t *pui8Size)
{
    uint8_t ui8CheckSum;
    uint8_t ui8Size;

    //
    // Get the size and the checksum.
    //
    do
    {
        if(UARTReceiveData(&ui8Size, 1))
        {
            return(-1);
        }
    }
    while(ui8Size == 0);

    if(UARTReceiveData(&ui8CheckSum, 1))
    {
        return(-1);
    }
    *pui8Size = ui8Size - 2;

    if(UARTReceiveData(pui8Data, *pui8Size))
    {
        *pui8Size = 0;
        return(-1);
    }

    //
    // Calculate the checksum from the data.
    //
    if(CheckSum(pui8Data, *pui8Size) != ui8CheckSum)
    {
        *pui8Size = 0;
        return(NakPacket());
    }

    return(AckPacket());
}

//*****************************************************************************
//
//! CheckSum() Calculates an 8 bit checksum
//!
//! \param pui8Data is a pointer to an array of 8 bit data of size ui8Size.
//! \param ui8Size is the size of the array that will run through the checksum
//!     algorithm.
//!
//! This function simply calculates an 8 bit checksum on the data passed in.
//!
//! \return The function returns the calculated checksum.
//
//*****************************************************************************
uint8_t
CheckSum(uint8_t *pui8Data, uint8_t ui8Size)
{
    int32_t i;
    uint8_t ui8CheckSum;

    ui8CheckSum = 0;

    for(i = 0; i < ui8Size; ++i)
    {
        ui8CheckSum += pui8Data[i];
    }
    return(ui8CheckSum);
}

//*****************************************************************************
//
//! SendPacket() sends a data packet.
//!
//! \param pui8Data is the location of the data to be sent to the device.
//! \param ui8Size is the number of bytes to send from puData.
//! \param bAck is a boolean that is true if an ACK/NAK packet should be
//! received in response to this packet.
//!
//! This function sends a packet of data to the device.
//!
//! \returns The function returns zero to indicated success while any non-zero
//!     value indicates a failure.
//
//*****************************************************************************
int32_t
SendPacket(uint8_t *pui8Data, uint8_t ui8Size, bool bAck)
{
    uint8_t ui8CheckSum;
    uint8_t ui8Ack;

    ui8CheckSum = CheckSum(pui8Data, ui8Size);

    //
    // Make sure that we add the bytes for the size and checksum to the total.
    //
    ui8Size += 2;

    //
    // Send the Size in bytes.
    //
    if(UARTSendData(&ui8Size, 1))
    {
        return(-1);
    }

    //
    // Send the CheckSum
    //
    if(UARTSendData(&ui8CheckSum, 1))
    {
        return(-1);
    }

    //
    // Now send the remaining bytes out.
    //
    ui8Size -= 2;

    //
    // Send the Data
    //
    if(UARTSendData(pui8Data, ui8Size))
    {
        return(-1);
    }

    //
    // Return immediately if no ACK/NAK is expected.
    //
    if(!bAck)
    {
        return(0);
    }

    //
    // Wait for the acknowledge from the device.
    //
    do
    {
        if(UARTReceiveData(&ui8Ack, 1))
        {
            return(-1);
        }
    }
    while(ui8Ack == 0);

    if(ui8Ack != COMMAND_ACK)
    {
        return(-1);
    }
    return(0);
}

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************
