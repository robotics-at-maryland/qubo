//*****************************************************************************
//
// CTS_Layer.h - Capacative Sense API Layer Header file definitions.
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

#ifndef __CTS_LAYER_H__
#define __CTS_LAYER_H__
//
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
// Values that indicate the current capsense event status in the
// global g_ui32CtsStatusReg variable.
//
//*****************************************************************************
#define EVNT                    0x00000001  // Event indicator bit.
#define PAST_EVNT               0x00000004  // Past event indicator bit.

//*****************************************************************************
//
// Bit-mask and values for indicating direction-of-interest in the global
// g_ui32CtsStatusReg variable.
//
//*****************************************************************************
#define DOI_MASK                0x00000002  // Direction-of-interest bit-mask
#define DOI_INC                 0x00000002  // Increasing direction of interest
#define DOI_DEC                 0x00000000  // Decreasing direction of interest

//*****************************************************************************
//
// Values used to set baseline capacitance tracking rates for changes in the
// direction-of-interest in g_ui32CtsStatusReg variable.
//
//*****************************************************************************
#define TRIDOI_VSLOW            0x00000000  // Very slow tracking
#define TRIDOI_SLOW             0x00000010  // Slow tracking
#define TRIDOI_MED              0x00000020  // Medium tracking
#define TRIDOI_FAST             0x00000030  // Fast tracking

//*****************************************************************************
//
// Values used to set baseline capacitance tracking rates for changes against
// the direction-of-interest in g_ui32CtsStatusReg variable.
//
//*****************************************************************************
#define TRADOI_FAST             0x00000000  // Fast tracking
#define TRADOI_MED              0x00000040  // Medium tracking
#define TRADOI_SLOW             0x00000080  // Slow tracking
#define TRADOI_VSLOW            0x000000C0  // Very slow tracking

//*****************************************************************************
//
// Prototypes for the APIs exported from CTS_Layer.c
//
//*****************************************************************************
extern void TI_CAPT_Init_Baseline(const tSensor*);
extern void TI_CAPT_Update_Baseline(const tSensor*, uint8_t);
extern void TI_CAPT_Reset_Tracking(void);
extern void TI_CAPT_Update_Tracking_DOI(bool);
extern void TI_CAPT_Update_Tracking_Rate(uint8_t);
extern void TI_CAPT_Raw(const tSensor*, uint32_t*);
extern void TI_CAPT_Custom(const tSensor *, uint32_t*);
extern uint8_t TI_CAPT_Button(const tSensor *);
extern const tCapTouchElement * TI_CAPT_Buttons(const tSensor *);
extern uint32_t TI_CAPT_Slider(const tSensor*);
extern uint32_t TI_CAPT_Wheel(const tSensor*);

extern uint32_t g_ui32Baselines[TOTAL_NUMBER_OF_ELEMENTS];

//*****************************************************************************
//
// Prototypes for static functions internal to CTS_Layer.c
//
//*****************************************************************************
uint8_t Dominant_Element (const tSensor*, uint32_t*);

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

#endif // __CTS_LAYER_H__
