//*****************************************************************************
//
// compdcm_task.h - Prototypes for the Complimentary Filtered DCM task.
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

#ifndef __CompDCM_TASK_H__
#define __CompDCM_TASK_H__

//*****************************************************************************
//
// The stack size for the display task.
//
//*****************************************************************************
#define COMPDCM_TASK_STACK_SIZE        1024         // Stack size in words

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xCompDCMHandle;

//*****************************************************************************
//
// Structure to hold the sensor data and provide access to other tasks.
//
//*****************************************************************************
typedef struct sCompDCMDataStruct
{
    //
    // boolean flag to indicate if the task is still actively updating data.
    //
    bool bActive;

    //
    // Array of Euler angles. Roll, Pitch, Yaw.
    //
    float pfEuler[3];

    //
    // Array of quaternion values
    //
    float pfQuaternion[4];

    //
    // Array of raw angular velocities from the sensor.
    //
    float pfAngularVelocity[3];

    //
    // Array of raw magnetic field strength sensor measurements.
    //
    float pfMagneticField[3];

    //
    // Array of raw accelerometer readings from the sensor.
    //
    float pfAcceleration[3];

    //
    // Tick counter time stamp at the most recent update of this struct.
    //
    portTickType xTimeStampTicks;

} sCompDCMData_t;

extern sCompDCMData_t g_sCompDCMData;


//*****************************************************************************
//
// Prototypes for the switch task.
//
//*****************************************************************************
extern uint32_t CompDCMTaskInit(void);
extern void CompDCMDataPrint(float *pfRPY, float *pfQuaternion);
extern uint32_t CompDCMDataEncodeJSON(char *pcBuf, uint32_t ui32BufSize);

#endif // __CompDCM_TASK_H__
