//*****************************************************************************
//
// tmp006_task.h - Prototypes for the TMP006 task.
//
// Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
//
//*****************************************************************************

#ifndef __TMP006_TASK_H__
#define __TMP006_TASK_H__

//*****************************************************************************
//
// The stack size for the display task.
//
//*****************************************************************************
#define TMP006_TASK_STACK_SIZE        768         // Stack size in words

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xTMP006Handle;

//*****************************************************************************
//
// Structure to hold the sensor data and provide access to other tasks.
//
//*****************************************************************************
typedef struct sTMP006DataStruct
{
    bool bActive;
    float fAmbient;
    float fObject;
    portTickType xTimeStampTicks;

} sTMP006Data_t;

extern sTMP006Data_t g_sTMP006Data;

//*****************************************************************************
//
// Prototypes for the switch task.
//
//*****************************************************************************
extern uint32_t TMP006TaskInit(void);
extern void TMP006DataPrint(float fAmbient, float fObject);
extern uint32_t TMP006DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize);

#endif // __TMP006_TASK_H__
