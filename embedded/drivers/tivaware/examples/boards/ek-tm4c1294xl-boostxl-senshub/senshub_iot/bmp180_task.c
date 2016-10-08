//*****************************************************************************
//
// bmp180_task.c - A simple task to read the sensor periodically.
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

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "inc/hw_gpio.h"
#include "driverlib/debug.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "drivers/buttons.h"
#include "drivers/pinout.h"
#include "sensorlib/hw_bmp180.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/bmp180.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "priorities.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "bmp180_task.h"
#include "tmp006_task.h"
#include "isl29023_task.h"
#include "compdcm_task.h"
#include "sht21_task.h"
#include "command_task.h"
#include "cloud_task.h"

//*****************************************************************************
//
// The I2C mutex
//
//*****************************************************************************
extern xSemaphoreHandle g_xI2CSemaphore;

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
extern tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Global variable to hold system clock speed returned from SysCtlClockFreqSet.
//
//*****************************************************************************
extern uint32_t g_ui32SysClock;

//*****************************************************************************
//
// Define BMP180 I2C Address.
//
//*****************************************************************************
#define BMP180_I2C_ADDRESS      0x77

//*****************************************************************************
//
// Global instance structure for the BMP180 sensor driver.
//
//*****************************************************************************
tBMP180 g_sBMP180Inst;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the I2C transaction to
// complete. Indicates that sensor data has been read and is now ready for
// processing.
//
//*****************************************************************************
xSemaphoreHandle g_xBMP180TransactionCompleteSemaphore;

//*****************************************************************************
//
// Global new error flag to store the error condition if encountered.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8BMP180I2CErrorStatus;

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
xTaskHandle g_xBMP180Handle;

//*****************************************************************************
//
// Global instance structure for the BMP180 sensor data to be published.
// This structure must be protected by the global g_xCloudDataSemaphore.
// The g_sCloudData struct will point to this structure for this sensor's data.
//
//*****************************************************************************
sBMP180Data_t g_sBMP180Data;

//*****************************************************************************
//
// BMP180 Sensor callback function.  Called at the end of BMP180 sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations and display. We
// also give back the I2C Semaphore so that other may use the I2C driver.
//
//*****************************************************************************
void
BMP180AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
{
    portBASE_TYPE xHigherPriorityTaskWokenTransaction;

    //
    // initialize the variable.
    //
    xHigherPriorityTaskWokenTransaction = pdFALSE;

    //
    // Give the binary semaphore to wake the task so it can resume
    // and check the status error code.
    //
    xSemaphoreGiveFromISR(g_xBMP180TransactionCompleteSemaphore,
                          &xHigherPriorityTaskWokenTransaction);

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8BMP180I2CErrorStatus = ui8Status;

    //
    // If a higher priority task was waiting for a semaphore released by this
    // isr then that high priority task will run when the ISR exits.
    //
    if(xHigherPriorityTaskWokenTransaction == pdTRUE)
    {
        portYIELD_FROM_ISR(true);
    }

}

//*****************************************************************************
//
// BMP180 Application error handler.
//
//*****************************************************************************
void
BMP180AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Take the UART semaphore.  Using the UART semaphore keeps all the parts
    // of the UART message from this task in the right order in the queue and
    // keeps others from injecting their messages between our message group.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1mBMP180 I2C Error: %d, File: %s, Line: %d\n",
               g_vui8BMP180I2CErrorStatus, pcFilename, ui32Line);

    //
    // Tell users where to find more info and return to normal color.
    //
    UARTprintf("See I2C status definitions in utils\\i2cm_drv.h\n\033[0m");

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);

    //
    // Change the active flag  to false and set the time stamp.  Alerts the
    // other tasks that this sensor is no longer active.
    //
    xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
    g_sBMP180Data.bActive = false;
    g_sBMP180Data.xTimeStampTicks = xTaskGetTickCount();
    xSemaphoreGive(g_xCloudDataSemaphore);

    //
    // Since we got an I2C error we will suspend this task and let other
    // tasks continue.  We will never again execute unless some other task
    // calls vTaskResume on us.
    //
    vTaskSuspend(NULL);
}

//*****************************************************************************
//
// This function will encode the BMP180 data into a JSON format string.
//
// \param pcBuf is a pointer to a buffer where the JSON string is stored.
// \param ui32BufSize is the size of the buffer pointer to by pcBuf.
//
// \return the number of bytes written to pcBuf as indicated by usnprintf.
//
//*****************************************************************************
uint32_t
BMP180DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize)
{
    uint32_t ui32SpaceUsed;
    char pcPressureBuf[12];
    char pcTemperatureBuf[12];
    char pcAltitudeBuf[12];

    //
    // Convert the float members of the struct into strings.
    //
    uftostr(pcPressureBuf, 12, 3, g_sBMP180Data.fPressure);
    uftostr(pcTemperatureBuf, 12, 3, g_sBMP180Data.fTemperature);
    uftostr(pcAltitudeBuf, 12, 3, g_sBMP180Data.fAltitude);

    //
    // Merge the float value strings into JSON formated string in the buffer.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "{\"sBMP180Data_t\":{"
                              "\"bActive\":%d,\"fPressure\":%s,"
                              "\"fTemperature\":%s,\"fAltitude\":%s}}",
                              g_sBMP180Data.bActive, pcPressureBuf,
                              pcTemperatureBuf, pcAltitudeBuf);

    //
    // Return size of string created.
    //
    return ui32SpaceUsed;
}

//*****************************************************************************
//
// BMP180 Data print function.  Takes the float versions of sensor data and
// prints them to the UART in a pretty format.
//
// This function manages the UART semaphore.  Calling function should manage
// the CloudData Semaphore
//
//*****************************************************************************
void
BMP180DataPrint(float fPressure, float fTemperature, float fAltitude)
{
    char pcPressureBuf[12];
    char pcTemperatureBuf[12];
    char pcAltitudeBuf[12];

    //
    // Convert the float members of the struct into strings.
    //
    uftostr(pcPressureBuf, 12, 3, fPressure);
    uftostr(pcTemperatureBuf, 12, 3, fTemperature);
    uftostr(pcAltitudeBuf, 12, 3, fAltitude);

    //
    // Take the UART semaphore.  Using the UART semaphore keeps all the parts
    // of the UART message from this task in the right order in the queue and
    // keeps others from injecting their messages between our message group.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print temperature with three digits of decimal precision.
    //
    UARTprintf("BMP180:\t\tTemp:\t\t%s\t", pcTemperatureBuf);


    //
    // Print Pressure with three digits of decimal precision.
    //
    UARTprintf("Pressure:\t%s\n", pcPressureBuf);

    //
    // Print altitude with three digits of decimal precision.
    //
    UARTprintf("BMP180:\t\tAltitude:\t%s\n", pcAltitudeBuf);

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);

}

//*****************************************************************************
//
// This task reads the BMP180 Pressure sensor and places the information in a
// shared structure.
//
//*****************************************************************************
static void
BMP180Task(void *pvParameters)
{
    portTickType xLastWakeTime;
    float fAltitude, fPressure, fTemperature;

    //
    // The binary semaphore is created full so we empty it first so we can
    // use it to wait for the AppCallback function.
    //
    xSemaphoreTake(g_xBMP180TransactionCompleteSemaphore, 0);

    //
    // Take the I2C semaphore so we can initialize the sensor. Keep it until
    // all init is complete for this sensor.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // BMP180 appears to need about 30 micro seconds of idle on the I2C bus
    // before a command is sent to it.
    //
    SysCtlDelay((g_ui32SysClock / 3000000) * 40);
    
    //
    // Initialize the BMP180 sensor.
    //
    BMP180Init(&g_sBMP180Inst, &g_sI2CInst, BMP180_I2C_ADDRESS,
               BMP180AppCallback, &g_sBMP180Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xBMP180TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so others can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8BMP180I2CErrorStatus)
    {
        BMP180AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Get the current time as a reference to start our delays.
    //
    xLastWakeTime = xTaskGetTickCount();

    //
    // Loop forever.
    //
    while(1)
    {
        //
        // Wait for the required amount of time to check back.
        //
        vTaskDelayUntil(&xLastWakeTime, BMP180_TASK_PERIOD_MS /
                                        portTICK_RATE_MS);

        //
        // Take the I2C semaphore.
        //
        xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);
        
        //
        // BMP180 appears to need about 30 micro seconds of idle on the I2C bus
        // before a command is sent to it.
        //
        SysCtlDelay((g_ui32SysClock / 3000000) * 40);
    
        //
        // Start a read of data from the pressure sensor. BMP180AppCallback is
        // called when the read is complete.
        //
        BMP180DataRead(&g_sBMP180Inst, BMP180AppCallback, &g_sBMP180Inst);

        //
        // Wait for the I2C Driver to tell us that transaction is complete.
        //
        xSemaphoreTake(g_xBMP180TransactionCompleteSemaphore, portMAX_DELAY);

        //
        // Give back the I2C Semaphore so other can use the I2C interface.
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // If an error occurred call the error handler immediately.
        //
        if(g_vui8BMP180I2CErrorStatus)
        {
            BMP180AppErrorHandler(__FILE__, __LINE__);
        }

        //
        // Get a local copy of the latest temperature and pressure data in
        // float format.
        //
        BMP180DataTemperatureGetFloat(&g_sBMP180Inst, &fTemperature);
        BMP180DataPressureGetFloat(&g_sBMP180Inst, &fPressure);

        //
        // Calculate the altitude.
        //
        fAltitude = 44330.0f * (1.0f - powf(fPressure / 101325.0f,
                                            1.0f / 5.255f));

        //
        // Publish the data to the global structure for consumption by other
        // tasks.
        //
        xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
        g_sBMP180Data.fTemperature = fTemperature;
        g_sBMP180Data.fPressure = fPressure;
        g_sBMP180Data.fAltitude = fAltitude;
        g_sBMP180Data.xTimeStampTicks = xTaskGetTickCount();
        xSemaphoreGive(g_xCloudDataSemaphore);

    }
}

//*****************************************************************************
//
// Initializes the BMP180 task.
//
//*****************************************************************************
uint32_t
BMP180TaskInit(void)
{
    //
    // Create binary semaphore for flow control of the task synchronizing with
    // the I2C data read complete ISR (callback).
    //
    vSemaphoreCreateBinary(g_xBMP180TransactionCompleteSemaphore);

    //
    // Create the switch task.
    //
    if(xTaskCreate(BMP180Task, (const portCHAR *)"BMP180",
                   BMP180_TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY +
                   PRIORITY_BMP180_TASK, g_xBMP180Handle) != pdTRUE)
    {
        //
        // Task creation failed.
        //
        return(1);
    }

    //
    // Check if Semaphore creation was successful.
    //
    if(g_xBMP180TransactionCompleteSemaphore == NULL)
    {
        //
        // Semaphore was not created successfully.
        //
        return(1);
    }

    //
    // Set the active flag that shows this task is running properly.
    // Clear other variables in the data struct. Semaphore not required here
    // scheduler is not yet running still in single thread/task mode.
    //
    g_sBMP180Data.bActive = true;
    g_sBMP180Data.fAltitude = 0.0f;
    g_sBMP180Data.fPressure = 0.0f;
    g_sBMP180Data.fTemperature = 0.0f;
    g_sBMP180Data.xTimeStampTicks = 0;

    //
    // Initialize the CloudData pointer to point to our BMP180 data which is
    // managed by this task.
    //
    g_sCloudData.psBMP180Data = &g_sBMP180Data;

    //
    // Success.
    //
    return(0);
}
