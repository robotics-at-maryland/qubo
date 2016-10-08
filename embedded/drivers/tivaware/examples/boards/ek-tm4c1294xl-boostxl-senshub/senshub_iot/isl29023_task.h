//*****************************************************************************
//
// isl29023_task.h - Prototypes for the ISL29023 task.
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

#ifndef __ISL29023_TASK_H__
#define __ISL29023_TASK_H__

//*****************************************************************************
//
// The stack size for the ISL29023 task.
//
//*****************************************************************************
#define ISL29023_TASK_STACK_SIZE        512         // Stack size in words

//*****************************************************************************
//
// Period in milliseconds to determine time between ISL29023 samples.
//
//*****************************************************************************
#define ISL29023_TASK_PERIOD_MS         1000       // periodic rate of the task

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xISL29023Handle;

//*****************************************************************************
//
// Structure to hold the sensor data and provide access to other tasks.
//
//*****************************************************************************
typedef struct sISL29023DataStruct
{
    //
    // boolean flag to indicate if task is still actively updating data.
    //
    bool bActive;

    //
    // Most recent visible light lux measurement.
    //
    float fVisible;

    //
    // Most recent infrared sprectrum light measurement. if implemented.
    //
    float fInfrared;

    //
    // The current range setting of the sensor.  An integer. See sensor
    // datasheet
    //
    uint8_t ui8Range;

    //
    // Tick counter timestamp at time of most recent update to the struct.
    //
    portTickType xTimeStampTicks;

} sISL29023Data_t;

extern sISL29023Data_t g_sISL29023Data;

//*****************************************************************************
//
// Prototypes for the ISL29023 task.
//
//*****************************************************************************
extern uint32_t ISL29023TaskInit(void);
extern void ISL29023DataPrint(float fVisible);
extern uint32_t ISL29023DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize);


#endif // __ISL29023_TASK_H__
