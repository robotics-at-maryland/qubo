//*****************************************************************************
//
// sht21_task.c - A simple task to read the sensor periodically.
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
#include "sensorlib/hw_sht21.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/sht21.h"
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
#include "sht21_task.h"
#include "compdcm_task.h"
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
// Define SHT21 I2C Address.
//
//*****************************************************************************
#define SHT21_I2C_ADDRESS      0x40

//*****************************************************************************
//
// Global instance structure for the SHT21 sensor driver.
//
//*****************************************************************************
tSHT21 g_sSHT21Inst;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the I2C transaction to
// complete. Sensor data has been read and is now ready for processing.
//
//*****************************************************************************
xSemaphoreHandle g_xSHT21TransactionCompleteSemaphore;

//*****************************************************************************
//
// Global new error flag to store the error condition if encountered.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8SHT21I2CErrorStatus;

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
xTaskHandle g_xSHT21Handle;

//*****************************************************************************
//
// Global instance structure for the SHT21 sensor data to be published.
//
//*****************************************************************************
sSHT21Data_t g_sSHT21Data;

//*****************************************************************************
//
// SHT21 Sensor callback function.  Called at the end of SHT21 sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations.
//
//*****************************************************************************
void
SHT21AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
{
    portBASE_TYPE xHigherPriorityTaskWokenTransaction;

    //
    // Init the local variable.
    //
    xHigherPriorityTaskWokenTransaction = pdFALSE;

    //
    // Give the binary semaphore to wake the task so it can resume
    // and check the status error code.
    //
    xSemaphoreGiveFromISR(g_xSHT21TransactionCompleteSemaphore,
                          &xHigherPriorityTaskWokenTransaction);

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8SHT21I2CErrorStatus = ui8Status;

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
// SHT21 Application error handler.
//
//*****************************************************************************
void
SHT21AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1mSHT21 I2C Error: %d, File: %s, Line: %d\n",
               g_vui8SHT21I2CErrorStatus, pcFilename, ui32Line);

    //
    // Tell users where to get more information and return terminal color
    // to normal.
    //
    UARTprintf("See I2C status definitions in utils\\i2cm_drv.h\n\033[0m");

    //
    // Give back the UART Semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);

    //
    // Change the active flag in the cloud data struct to show this sensor
    // is no longer being actively updated.
    //
    xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
    g_sSHT21Data.bActive = false;
    g_sSHT21Data.xTimeStampTicks = xTaskGetTickCount();
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
// This function will encode the sensor data into a JSON format string.
//
// \param pcBuf is a pointer to a buffer where the JSON string is stored.
// \param ui32BufSize is the size of the buffer pointer to by pcBuf.
//
// \return the number of bytes written to pcBuf as indicated by usnprintf.
//
//*****************************************************************************
uint32_t
SHT21DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize)
{
    uint32_t ui32SpaceUsed;
    char pcTemperatureBuf[12];
    char pcHumidityBuf[12];

    //
    // Convert floating point members of the struct into strings.
    //
    uftostr(pcTemperatureBuf, 12, 3, g_sSHT21Data.fTemperature);
    uftostr(pcHumidityBuf, 12, 3, g_sSHT21Data.fHumidity);

    //
    // Merge the strings into the buffer in the JSON format.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "{\"sSHT21Data_t\":{"
                              "\"bActive\":%d,\"fTemperature\":%s,"
                              "\"fHumidity\":%s}}", g_sSHT21Data.bActive,
                              pcTemperatureBuf, pcHumidityBuf);

    //
    // Return size of string created.
    //
    return ui32SpaceUsed;
}

//*****************************************************************************
//
// SHT21 Data print function.  Takes the float versions of sensor data and
// prints them to the UART in a pretty format.
//
//*****************************************************************************
void SHT21DataPrint(float fHumidity, float fTemperature)
{
    char pcTemperatureBuf[12];
    char pcHumidityBuf[12];

    //
    // Convert floating point members of the struct into strings.
    //
    uftostr(pcTemperatureBuf, 12, 3, fTemperature);
    uftostr(pcHumidityBuf, 12, 3, fHumidity);

    //
    // Take the UART semaphore to keep this tasks messages together in the
    // queue.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print temperature with three digits of decimal precision.
    //
    UARTprintf("SHT21:\t\tTemp:\t\t%s\t", pcTemperatureBuf);

    //
    // Print Humidity with three digits of decimal precision.
    //
    UARTprintf("Humidity:\t%s\n", pcHumidityBuf);

    //
    // Give back the UART Semaphore let other tasks send/receive.
    //
    xSemaphoreGive(g_xUARTSemaphore);

}

//*****************************************************************************
//
// Send the commands for the SHT21 to perform the humidity measurement.
// return the result as a float.
//
//*****************************************************************************
float
SHT21MeasureHumidity(void)
{
    portTickType xLastWakeTime;
    float fHumidity;

    //
    // Take the I2C semaphore.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Write the command to start a humidity measurement
    //
    SHT21Write(&g_sSHT21Inst, SHT21_CMD_MEAS_RH, g_sSHT21Inst.pui8Data, 0,
               SHT21AppCallback, &g_sSHT21Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xSHT21TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8SHT21I2CErrorStatus)
    {
        SHT21AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Get the current time as a reference to start our delays.
    //
    xLastWakeTime = xTaskGetTickCount();

    //
    // Wait for about 33 milliseconds.  Datasheet claims measurement takes
    // 29 milliseconds.
    //
    vTaskDelayUntil(&xLastWakeTime, 33 / portTICK_RATE_MS);

    //
    // Take the I2C semaphore.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Start a read of data from the pressure sensor. SHT21AppCallback is
    // called when the read is complete.
    //
    SHT21DataRead(&g_sSHT21Inst, SHT21AppCallback, &g_sSHT21Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xSHT21TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8SHT21I2CErrorStatus)
    {
        SHT21AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Get the latest humidity reading in float format to our local variable.
    //
    SHT21DataHumidityGetFloat(&g_sSHT21Inst, &fHumidity);

    return(fHumidity);

}

//*****************************************************************************
//
// Send the commands for the SHT21 to perform the temperature measurement.
// return the result as a float.
//
//*****************************************************************************
float
SHT21MeasureTemperature(void)
{
    portTickType xLastWakeTime;
    float fTemperature;

    //
    // Take the I2C semaphore.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Write the command to start a temperature measurement
    //
    SHT21Write(&g_sSHT21Inst, SHT21_CMD_MEAS_T, g_sSHT21Inst.pui8Data, 0,
               SHT21AppCallback, &g_sSHT21Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xSHT21TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8SHT21I2CErrorStatus)
    {
        SHT21AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Get the current time as a reference to start our delays.
    //
    xLastWakeTime = xTaskGetTickCount();

    //
    // Wait for about 90 milliseconds.  Datasheet claims measurement takes
    // 85 milliseconds.
    //
    vTaskDelayUntil(&xLastWakeTime, 90 / portTICK_RATE_MS);

    //
    // Take the I2C semaphore.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Start a read of data from the pressure sensor. SHT21AppCallback is
    // called when the read is complete.
    //
    SHT21DataRead(&g_sSHT21Inst, SHT21AppCallback, &g_sSHT21Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xSHT21TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8SHT21I2CErrorStatus)
    {
        SHT21AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Get the temperature reading in float format to our local variable.
    //
    SHT21DataTemperatureGetFloat(&g_sSHT21Inst, &fTemperature);

    //
    // Return the result.
    //
    return(fTemperature);

}

//*****************************************************************************
//
// This task reads the SHT21 sensor and places the information in a
// shared structure.
//
//*****************************************************************************
static void
SHT21Task(void *pvParameters)
{
    portTickType xLastWakeTime;
    float fHumidity, fTemperature;

    //
    // The binary semaphore is created full so we empty it first so we can
    // use it to wait for the AppCallback function.
    //
    xSemaphoreTake(g_xSHT21TransactionCompleteSemaphore, 0);

    //
    // Take the I2C semaphore so we can init the sensor. Keep it until all init
    // is complete for this sensor.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Initialize the SHT21 sensor.
    //
    SHT21Init(&g_sSHT21Inst, &g_sI2CInst, SHT21_I2C_ADDRESS,
               SHT21AppCallback, &g_sSHT21Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xSHT21TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8SHT21I2CErrorStatus)
    {
        SHT21AppErrorHandler(__FILE__, __LINE__);
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
        vTaskDelayUntil(&xLastWakeTime, SHT21_TASK_PERIOD_MS /
                                        portTICK_RATE_MS);

        //
        // Get a local copy of the latest temperature and pressure data in
        // float format.
        //
        fHumidity = SHT21MeasureHumidity();
        fTemperature = SHT21MeasureTemperature();

        //
        // Publish the data to the global structure for consumption by other
        // tasks.
        //
        xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
        g_sSHT21Data.fTemperature = fTemperature;
        g_sSHT21Data.fHumidity = fHumidity;
        g_sSHT21Data.xTimeStampTicks = xTaskGetTickCount();
        xSemaphoreGive(g_xCloudDataSemaphore);

    }
}

//*****************************************************************************
//
// Initializes the switch task.
//
//*****************************************************************************
uint32_t
SHT21TaskInit(void)
{
    //
    // Create binary semaphore for flow control of the task synchronizing with
    // the I2C data read complete ISR (callback).
    //
    vSemaphoreCreateBinary(g_xSHT21TransactionCompleteSemaphore);

    //
    // Create the switch task.
    //
    if(xTaskCreate(SHT21Task, (const portCHAR *)"SHT21     ",
                   SHT21_TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY +
                   PRIORITY_SHT21_TASK, g_xSHT21Handle) != pdTRUE)
    {
        //
        // Task creation failed.
        //
        return(1);
    }

    //
    // Check if Semaphore creation was successfull.
    //
    if(g_xSHT21TransactionCompleteSemaphore == NULL)
    {
        //
        // Semaphore was not created successfully.
        //
        return(1);
    }

    //
    // Set the active flag that shows this task is running properly.
    //
    g_sSHT21Data.bActive = true;
    g_sSHT21Data.fHumidity = 0.0f;
    g_sSHT21Data.fTemperature = 0.0f;
    g_sSHT21Data.xTimeStampTicks = 0;

    //
    // Set the pointer in the cloud data structure to point to the local
    // data from this sensor.
    //
    g_sCloudData.psSHT21Data = &g_sSHT21Data;

    //
    // Success.
    //
    return(0);
}
