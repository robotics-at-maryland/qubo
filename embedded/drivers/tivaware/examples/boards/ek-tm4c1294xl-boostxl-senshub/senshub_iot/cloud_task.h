//*****************************************************************************
//
// cloud_task.h - Task to connect and communicate to the cloud server. This
// task also manages board level user switch and LED function for this app.
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

#ifndef __CLOUD_TASK_H__
#define __CLOUD_TASK_H__

//*****************************************************************************
//
// Recognized requests to be sent to the cloud task.
//
//*****************************************************************************
#define CLOUD_REQUEST_SYNC          1       // Sync with Exosite now.
#define CLOUD_REQUEST_ACTIVATE      2       // Obtain a new CIK (re-provision)
#define CLOUD_REQUEST_PROXY_SET     3       // Set or "off" the proxy
#define CLOUD_REQUEST_START         4       // Start or re-start connection

//*****************************************************************************
//
// A structure definition to pass requests to the cloud task from other tasks.
// The ui32Request element should be one of the defined requests above.
// The buffer can be used for arguments, such as the proxy URL and port.
//
//*****************************************************************************
typedef struct sCloudTaskRequest
{
    //
    // The request identifier for this message.
    //
    uint32_t ui32Request;

    //
    // A message buffer to hold additional message data.
    //
    char pcBuf[128];
}sCloudTaskRequest_t;

//*****************************************************************************
//
// The stack size for the cloud task.
//
//*****************************************************************************
#define CLOUD_TASK_STACK_SIZE        4096         // Stack size in words

//*****************************************************************************
//
// Period in milliseconds to determine time between cloud updates.
//
//*****************************************************************************
#define CLOUD_TIMER_PERIOD_MS         10000       // periodic rate of the timer

//*****************************************************************************
//
// Period in milliseconds to determine rate of polling users switches.
//
//*****************************************************************************
#define SWITCH_TIMER_PERIOD_MS        8        // periodic rate of the timer

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xCloudTaskHandle;

//*****************************************************************************
//
// A global queue so tasks can send requests to this task.
//
//*****************************************************************************
extern xQueueHandle g_xCloudTaskRequestQueue;

//*****************************************************************************
//
// Number of requests that the queue can hold.
//
//*****************************************************************************
#define CLOUD_QUEUE_LENGTH      32

//*****************************************************************************
//
// Size of each item in the queue in bytes.
//
//*****************************************************************************
#define CLOUD_QUEUE_ITEM_SIZE   (sizeof(sCloudTaskRequest_t))

//*****************************************************************************
//
// Semaphore used to control access to the Cloud Data structure.
//
//*****************************************************************************
extern xSemaphoreHandle g_xCloudDataSemaphore;

//*****************************************************************************
//
// Struct typedef to hold the board level system data.
//
//*****************************************************************************
typedef struct sBoardDataStruct
{
    //
    // Array to hold the current state of the four LEDs
    //
    uint8_t pui8LED[4];

    //
    // Array to hold the number of times each of the two buttons has been
    // pressed.
    //
    uint32_t ui32SwitchPressCount[2];

}sBoardData_t;

//*****************************************************************************
//
// Define the max number of expected tasks to size the TaskStatus Array.
//
//*****************************************************************************
#define TASK_STATUS_ARRAY_SIZE  16

//*****************************************************************************
//
// Struct typedef to hold the FreeRTOS task statistics.
//
//*****************************************************************************
typedef struct sTaskStatistics
{
    //
    // Array to hold task statistic data provided by FreeRTOS.
    //
    xTaskStatusType pxTaskStatus[TASK_STATUS_ARRAY_SIZE];

    //
    // Percentage of total run time when each task was actually running.
    //
    float pfCPUUsage[TASK_STATUS_ARRAY_SIZE];
    //
    // Actual number of currently active tasks with data in the array.
    //
    uint32_t ui32NumActiveTasks;

    //
    // Total run time of all tasks combined measured in RTOS Ticks.
    //
    unsigned long ulTotalRunTime;

}sTaskStatistics_t;

//*****************************************************************************
//
// Structure that binds together all the data types that will be published to
// the cloud provider. Each of these sub structures is owned and managed by
// the associated task.
//
// Each task must use the CloudDataSemaphore to guard the data when they write
// data to their individual data structure.
//
// The cloud task must use the CloudDataSemaphore to read the data from this
// structure. This makes sure that data is valid and we don't have concurrent
// access issues.
//
//*****************************************************************************
typedef struct sCloudDataStruct
{
    //
    // Pointer to general system data such as LEDs and Switches.
    //
    sBoardData_t *psBoardData;

    //
    // Pointer to the TMP006 infrared contact-less temperature data structure.
    //
    sTMP006Data_t *psTMP006Data;

    //
    // Pointer to the SHT21 humidity sensor data.
    //
    sSHT21Data_t *psSHT21Data;

    //
    // Pointer to the complimentary direct cosine matrix data structure. This
    // is provided by the MPU9150 sensor and processed by the compdcm_task.
    //
    sCompDCMData_t *psCompDCMData;

    //
    // Pointer to the BMP180 pressure sensor data structure.
    //
    sBMP180Data_t *psBMP180Data;

    //
    // Pointer to the ISL29023 light sensor data type.
    //
    sISL29023Data_t *psISL29023Data;

    //
    // Pointer to FreeRTOS task statistic information.
    //
    sTaskStatistics_t *psTaskStatistics;

} sCloudData_t;

extern sCloudData_t g_sCloudData;

//*****************************************************************************
//
// Prototypes for the cloud task.
//
//*****************************************************************************
extern uint32_t CloudTaskInit(void);
extern void CloudTaskStatsPrint(void);
extern uint32_t uftostr(char * pcStr, uint32_t ui32Size,
                        uint32_t ui32Precision, float fValue);
extern bool CloudProvisionExosite(void);

#endif // __CLOUD_TASK_H__
