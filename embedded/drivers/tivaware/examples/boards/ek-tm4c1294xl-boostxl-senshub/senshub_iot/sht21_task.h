//*****************************************************************************
//
// sht21_task.h - Prototypes for the SHT21 sensor task.
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

#ifndef __SHT21_TASK_H__
#define __SHT21_TASK_H__

//*****************************************************************************
//
// The stack size for the display task.
//
//*****************************************************************************
#define SHT21_TASK_STACK_SIZE        512         // Stack size in words

//*****************************************************************************
//
// Period in milliseconds to determine time between SHT21 samples.
//
//*****************************************************************************
#define SHT21_TASK_PERIOD_MS         1000        // periodic rate of the task

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xSHT21Handle;

//*****************************************************************************
//
// Structure to hold the sensor data and provide access to other tasks.
//
//*****************************************************************************
typedef struct sSHT21DataStruct
{
    //
    // Boolean flag to indicate if the task is actively updating the struct.
    //
    bool bActive;

    //
    // Most recent temperature measurement from this sensor.
    //
    float fTemperature;

    //
    // Most recent humidity measurement from this sensor.
    //
    float fHumidity;

    //
    // Timestamp in RTOS ticks of the most recent update.
    //
    portTickType xTimeStampTicks;

} sSHT21Data_t;

extern sSHT21Data_t g_sSHT21Data;

//*****************************************************************************
//
// Prototypes for the task.
//
//*****************************************************************************
extern uint32_t SHT21TaskInit(void);
extern void SHT21DataPrint(float fHumidity, float fTemperature);
extern uint32_t SHT21DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize);

#endif // __SHT21_TASK_H__
