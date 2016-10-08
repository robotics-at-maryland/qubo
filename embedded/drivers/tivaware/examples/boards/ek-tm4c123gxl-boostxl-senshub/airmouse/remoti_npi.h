//*****************************************************************************
//
// remoti_npi.h - Network Processor Interface defines and declarations.
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

#ifndef __REMOTI_NPI_H__
#define __REMOTI_NPI_H__

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
// NPI (Network Processor Interface) constants
//
//*****************************************************************************
#define NPI_SYNC_MSG_TIMEOUT    10
#define NPI_MAX_DATA_LEN        256

//*****************************************************************************
//
// RemoTI Message format.
//
//*****************************************************************************
typedef struct
{
    //
    // Start of frame indicator.
    //
    uint8_t ui8SOF;

    //
    // Message length.
    //
    uint8_t ui8Length;

    //
    // RemoTI Subsystem identifier
    //
    uint8_t ui8SubSystem;

    //
    // RemoTI command identifier
    //
    uint8_t ui8CommandID;

    //
    // Data payload of the packet.
    //
    uint8_t pui8Data[NPI_MAX_DATA_LEN];
} tRemoTIMsg;

//*****************************************************************************
//
// NPI Functions exported by this driver.
//
//*****************************************************************************
extern void NPI_Init(tRemoTICallback *pfnCallback);
extern void NPI_MsgWrap(tRemoTIMsg* pMsg);
extern void NPI_SendAsynchData(tRemoTIMsg *pMsg);
extern bool NPI_SendSynchData(tRemoTIMsg *pMsg);
extern void NPI_MsgRxCallback(uint32_t ui32Command0);

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

#endif // __REMOTI_NPI_H__
