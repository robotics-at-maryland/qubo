//*****************************************************************************
//
// remoti_rtis.h - RemoTI API Surrogate defines and declarations.
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

#ifndef __REMOTI_RTIS_H__
#define __REMOTI_RTIS_H__

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
// RTIS Command Identifiers for NPI Callback.
//
//*****************************************************************************
#define RTIS_CMD_ID_RTI_READ_ITEM                                             \
                                0x01
#define RTIS_CMD_ID_RTI_WRITE_ITEM                                            \
                                0x02
#define RTIS_CMD_ID_RTI_INIT_REQ                                              \
                                0x03
#define RTIS_CMD_ID_RTI_PAIR_REQ                                              \
                                0x04
#define RTIS_CMD_ID_RTI_SEND_DATA_REQ                                         \
                                0x05
#define RTIS_CMD_ID_RTI_ALLOW_PAIR_REQ                                        \
                                0x06
#define RTIS_CMD_ID_RTI_STANDBY_REQ                                           \
                                0x07
#define RTIS_CMD_ID_RTI_RX_ENABLE_REQ                                         \
                                0x08
#define RTIS_CMD_ID_RTI_ENABLE_SLEEP_REQ                                      \
                                0x09
#define RTIS_CMD_ID_RTI_DISABLE_SLEEP_REQ                                     \
                                0x0A
#define RTIS_CMD_ID_RTI_UNPAIR_REQ                                            \
                                0x0B
#define RTIS_CMD_ID_RTI_PAIR_ABORT_REQ                                        \
                                0x0C
#define RTIS_CMD_ID_RTI_ALLOW_PAIR_ABORT_REQ                                  \
                                0x0D
#define RTIS_CMD_ID_TEST_PING_REQ                                             \
                                0x10
#define RTIS_CMD_ID_RTI_TEST_MODE_REQ                                         \
                                0x11
#define RTIS_CMD_ID_RTI_RX_COUNTER_GET_REQ                                    \
                                0x12
#define RTIS_CMD_ID_RTI_SW_RESET_REQ                                          \
                                0x13
#define RTIS_CMD_ID_RTI_READ_ITEM_EX                                          \
                                0x21
#define RTIS_CMD_ID_RTI_WRITE_ITEM_EX                                         \
                                0x22

//*****************************************************************************
//
// RTIS Confirmation Identifiers.
//
//*****************************************************************************
#define RTIS_CMD_ID_RTI_INIT_CNF                                              \
                                0x01
#define RTIS_CMD_ID_RTI_PAIR_CNF                                              \
                                0x02
#define RTIS_CMD_ID_RTI_SEND_DATA_CNF                                         \
                                0x03
#define RTIS_CMD_ID_RTI_ALLOW_PAIR_CNF                                        \
                                0x04
#define RTIS_CMD_ID_RTI_REC_DATA_IND                                          \
                                0x05
#define RTIS_CMD_ID_RTI_STANDBY_CNF                                           \
                                0x06
#define RTIS_CMD_ID_RTI_RX_ENABLE_CNF                                         \
                                0x07
#define RTIS_CMD_ID_RTI_ENABLE_SLEEP_CNF                                      \
                                0x08
#define RTIS_CMD_ID_RTI_DISABLE_SLEEP_CNF                                     \
                                0x09
#define RTIS_CMD_ID_RTI_UNPAIR_CNF                                            \
                                0x0A
#define RTIS_CMD_ID_RTI_UNPAIR_IND                                            \
                                0x0B
#define RTIS_CMD_ID_RTI_PAIR_ABORT_CNF                                        \
                                0x0C

//*****************************************************************************
//
// RTI States
//
//*****************************************************************************
#define RTIS_STATE_INIT         0
#define RTIS_STATE_READY        1
#define RTIS_STATE_NETWORK_LAYER_BRIDGE                                       \
                                2

//*****************************************************************************
//
// RTI Surrogate API function prototypes.
//
//*****************************************************************************
extern void RTI_Init(void);
extern uint_fast8_t RTI_ReadItemEx(uint8_t ui8ProfileID, uint8_t ui8ItemID,
                                   uint8_t ui8Length, uint8_t *pui8Value);
extern uint_fast8_t RTI_WriteItemEx(uint_fast8_t ui8ProfileID,
                                    uint_fast8_t ui8ItemID,
                                    uint_fast8_t ui8Length,
                                    const uint8_t *pui8Value);
extern void RTI_InitReq(void);
extern void RTI_PairReq(void);
extern void RTI_PairAbortReq(void);
extern void RTI_AllowPairReq(void);
extern void RTI_AllowPairAbortReq(void);
extern void RTI_SendDataReq(uint8_t ui8DestIndex, uint8_t ui8ProfileID,
                            uint16_t ui16VendorID, uint8_t ui8TXOptions,
                            uint8_t ui8Length, uint8_t *pui8Data);
extern void RTI_StandbyReq(uint8_t ui8Mode);
extern void RTI_RxEnableReq(uint32_t ui32Duration);
extern void RTI_DisableSleepReq(void);
extern void RTI_TestModeReq(uint8_t ui8Mode, int8_t i8TXPower,
                            uint8_t ui8Channel);
extern uint16_t RTI_TestRxCounterGetReq(uint8_t ui8ResetFlag);
extern void RTIS_AsynchMsgCallback(uint32_t ui32CallbackData);

//*****************************************************************************
//
// Write a configuration parameter to the RNP.
//
// \param ui8ItemId the identifier of the item to be written.
// \param ui8Length the number of bytes to write.
// \param pui8Value a pointer to the bytes to be written.
//
// This function is a convience wrapper around the \e RTI_WriteItemEx function.
// This function calls the extended version of the function with a default
// profile identifier.
//
// \return Returns \e RTI_SUCCESS or \e RTI_ERROR_NO_RESPONSE.
//
//*****************************************************************************
inline uint_fast8_t
RTI_WriteItem(uint8_t ui8ItemID, uint8_t ui8Length,
              const uint8_t *pui8Value)
{
    //
    // Call the extended version of the function with a default profile.
    //
    return RTI_WriteItemEx(RTI_PROFILE_RTI, ui8ItemID, ui8Length, pui8Value);
}

//*****************************************************************************
//
// Convenience wrapper function around the \e RTI_ReadItemEx function.
//
// \param ui8ItemId the identifier of the item to be read.
// \param ui8Length the length in bytes to be read.
// \param pui8Value storage location where the read result should be stored.
//
// This function adds a default profile ID onto the read item request and sends
// it to the \e RTI_ReadItemEx function.  If successful the read value will be
// stored at the location pointed to by pui8Value.
//
// \return Returns \e RTI_SUCCESS or \e RTI_ERROR_NO_RESPONSE.
//
//*****************************************************************************
inline uint_fast8_t
RTI_ReadItem(uint8_t ui8ItemID, uint8_t ui8Length,
             uint8_t *pui8Value)
{
    //
    // Call the extended Read Item function with the default RTI profile.
    //
    return RTI_ReadItemEx( RTI_PROFILE_RTI, ui8ItemID, ui8Length, pui8Value);
}

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

#endif // __REMOTI_RTIS_H__
