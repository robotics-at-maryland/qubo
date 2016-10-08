//*****************************************************************************
//
// bmp180_task.h - Prototypes for the BMP180 sensor task.
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

#ifndef __BMP180_TASK_H__
#define __BMP180_TASK_H__

//*****************************************************************************
//
// The stack size for the display task.
//
//*****************************************************************************
#define BMP180_TASK_STACK_SIZE        512         // Stack size in words

//*****************************************************************************
//
// Period in milliseconds to determine time between BMP180 samples.
//
//*****************************************************************************
#define BMP180_TASK_PERIOD_MS         1000        // periodic rate of the task

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xBMP180Handle;

//*****************************************************************************
//
// Structure to hold the sensor data and provide access to other tasks.
//
//*****************************************************************************
typedef struct sBMP180DataStruct
{
    //
    // boolean flag to indicate if this task is actively updating these data
    // fields.
    //
    bool bActive;

    //
    // The most recent pressure reading from the sensor.
    //
    float fPressure;

    //
    // The most recent altitude estimate. Calculated from pressure data.
    //
    float fAltitude;

    //
    // Most recent temperature reading from the sensor.
    //
    float fTemperature;

    //
    // Tick counter snapshot at the time of the most recent update.
    //
    portTickType xTimeStampTicks;

} sBMP180Data_t;

extern sBMP180Data_t g_sBMP180Data;

//*****************************************************************************
//
// Prototypes for the task.
//
//*****************************************************************************
extern uint32_t BMP180TaskInit(void);
extern void BMP180DataPrint(float fPressure, float fTemperature,
                                             float fAltitude);
extern uint32_t BMP180DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize);

#endif // __BMP180_TASK_H__
