//*****************************************************************************
//
// tmp006_task.c - A task to process the TMP006 contactless temperature sensor.
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
// This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"
#include "inc/hw_ints.h"
#include "driverlib/debug.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/rom.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "drivers/pinout.h"
#include "drivers/buttons.h"
#include "utils/ustdlib.h"
#include "utils/uartstdio.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/hw_tmp006.h"
#include "sensorlib/tmp006.h"
#include "priorities.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "tmp006_task.h"
#include "bmp180_task.h"
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
// Define TMP006 I2C Address.
//
//*****************************************************************************
#define TMP006_I2C_ADDRESS      0x41

//*****************************************************************************
//
// Global instance structure for the TMP006 sensor driver.
//
//*****************************************************************************
tTMP006 g_sTMP006Inst;

//*****************************************************************************
//
// Global instance structure for the TMP006 sensor data to be published.
//
//*****************************************************************************
sTMP006Data_t g_sTMP006Data;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the I2C transaction to
// complete. Sensor data has been read and is now ready for processing.
//
//*****************************************************************************
xSemaphoreHandle g_xTMP006TransactionCompleteSemaphore;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the GPIO interrupt which
// indicates that the sensor has data that needs to be read.
//
//*****************************************************************************
xSemaphoreHandle g_xTMP006DataReadySemaphore;

//*****************************************************************************
//
// Global new error flag to store the error condition if encountered.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8TMP006I2CErrorStatus;

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
xTaskHandle g_xTMP006Handle;

//*****************************************************************************
//
// TMP006 Sensor callback function.  Called at the end of TMP006 sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations and display. We
// also give back the I2C Semaphore so that other may use the I2C driver.
//
//*****************************************************************************
void
TMP006AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
{
    portBASE_TYPE xHigherPriorityTaskWokenTransaction;

    //
    // Initialize the task wake flag.
    //
    xHigherPriorityTaskWokenTransaction = pdFALSE;

    //
    // Let the task resume so that it can check the status flag.
    //
    xSemaphoreGiveFromISR(g_xTMP006TransactionCompleteSemaphore,
                          &xHigherPriorityTaskWokenTransaction);

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8TMP006I2CErrorStatus = ui8Status;

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
// TMP006 Application error handler.
//
//*****************************************************************************
void
TMP006AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1mError: %d, File: %s, Line: %d\n",
               g_vui8TMP006I2CErrorStatus, pcFilename, ui32Line);

    //
    // Tell users where to find more information and reset the terminal color.
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
    g_sTMP006Data.bActive = false;
    g_sTMP006Data.xTimeStampTicks = xTaskGetTickCount();
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
TMP006DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize)
{
    uint32_t ui32SpaceUsed;
    char pcAmbientBuf[12];
    char pcObjectBuf[12];

    //
    // Convert the floating point members of the struct to strings.
    //
    uftostr(pcAmbientBuf, 12, 3, g_sTMP006Data.fAmbient);
    uftostr(pcObjectBuf, 12, 3, g_sTMP006Data.fObject);

    //
    // Merge the strings into a single JSON formated string in the buffer.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "{\"sTMP006Data_t\":{"
                              "\"bActive\":%d,\"fAmbient\":%s,"
                              "\"fObject\":%s}}", g_sTMP006Data.bActive,
                              pcAmbientBuf, pcObjectBuf);

    //
    // Return size of string created.
    //
    return ui32SpaceUsed;
}

//*****************************************************************************
//
// TMP006 Data print function.  Takes the float versions of ambient and object
// temperature and prints them to the UART in a pretty format.
//
//*****************************************************************************
void
TMP006DataPrint(float fAmbient, float fObject)
{
    char pcAmbientBuf[12];
    char pcObjectBuf[12];

    //
    // Convert the floating point members of the struct to strings.
    //
    uftostr(pcAmbientBuf, 12, 3, fAmbient);
    uftostr(pcObjectBuf, 12, 3, fObject);

    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Send the data to the virtual com port buffer.
    //
    UARTprintf("TMP006:\t\tAmbient:\t%s\t", pcAmbientBuf);

    //
    // Send the data to the virtual com port buffer.
    //
    UARTprintf("Object:\t\t%s\n", pcObjectBuf);

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);
}

//*****************************************************************************
//
// Called by the NVIC as a result of GPIO port H interrupt event. For this
// application GPIO port H pin 2 is the interrupt line for the TMP006
//
// To use the sensor hub on BoosterPack 2 modify this function to accept and
// handle interrupts on GPIO Port P pin 5.  Also move the reference to this
// function in the startup file to GPIO Port P Int handler position in the
// vector table.
//
//*****************************************************************************
void
IntHandlerGPIOPortH(void)
{
    portBASE_TYPE xHigherPriorityTaskWoken;
    uint32_t ui32Status;

    //
    // Get the status register to determin which pin caused the interrupt.
    //
    ui32Status = GPIOIntStatus(GPIO_PORTH_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    GPIOIntClear(GPIO_PORTH_BASE, ui32Status);

    //
    // Verify that it was the TMP006 pin that caused this interrupt.
    //
    if(ui32Status & GPIO_PIN_2)
    {
        //
        // Give the TMP006 Data Ready Pin binary semaphore. This will
        // release the main TMP006 task so it can go get the data.
        //
        xHigherPriorityTaskWoken = pdFALSE;
        xSemaphoreGiveFromISR(g_xTMP006DataReadySemaphore,
                              &xHigherPriorityTaskWoken);

        //
        // If a higher priority task was waiting for this semaphore then this
        // call will make sure it runs immediately after the ISR returns.
        //
        if(xHigherPriorityTaskWoken == pdTRUE)
        {
            portYIELD_FROM_ISR(true);
        }
    }
}

//*****************************************************************************
//
// This task reads the TMP006 and places that data into a shared structure.
//
//*****************************************************************************
static void
TMP006Task(void *pvParameters)
{
    float fAmbient, fObject;

    //
    // The binary semaphore is created full so we take it up front and use it
    // later to sync between this task and the AppCallback function which is in
    // the I2C interrupt context. Likewise the GPIO port interrupt.
    //
    xSemaphoreTake(g_xTMP006TransactionCompleteSemaphore, 0);
    xSemaphoreTake(g_xTMP006DataReadySemaphore, 0);

    //
    // Take the I2C semaphore. Keep it until all init is complete for this
    // sensor.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // We have the mutex for I2C so do I2C Init of the TMP006 sensor.
    // TMP006AppCallback will give the mutex back from the ISR.
    //
    TMP006Init(&g_sTMP006Inst, &g_sI2CInst, TMP006_I2C_ADDRESS,
               TMP006AppCallback, &g_sTMP006Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xTMP006TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8TMP006I2CErrorStatus)
    {
        //
        // Give back the I2C Semaphore so others can use the I2C interface.
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // Call the error handler
        //
        TMP006AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Enable the DRDY pin indication that a conversion is in progress.
    //
    TMP006ReadModifyWrite(&g_sTMP006Inst, TMP006_O_CONFIG,
                          ~TMP006_CONFIG_EN_DRDY_PIN_M,
                          TMP006_CONFIG_EN_DRDY_PIN, TMP006AppCallback,
                          &g_sTMP006Inst);

    //
    // Wait for the I2C Driver to tell us that transaction is complete.
    //
    xSemaphoreTake(g_xTMP006TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so others can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8TMP006I2CErrorStatus)
    {
        TMP006AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Loop forever.
    //
    while(1)
    {
        //
        // Take the binary semaphore that alerts us that data ready on the
        // sensor.
        //
        xSemaphoreTake(g_xTMP006DataReadySemaphore, portMAX_DELAY);

        //
        // Take the I2C semaphore so we can go get the data.
        //
        xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

        //
        // Get the data from the sensor.
        //
        TMP006DataRead(&g_sTMP006Inst, TMP006AppCallback, &g_sTMP006Inst);

        //
        // Wait for the I2C Driver to tell us that transaction is complete.
        //
        xSemaphoreTake(g_xTMP006TransactionCompleteSemaphore, portMAX_DELAY);

        //
        // Give back the I2C Semaphore so others can use the I2C interface.
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // If an error occurred call the error handler immediately.
        //
        if(g_vui8TMP006I2CErrorStatus)
        {
            TMP006AppErrorHandler(__FILE__, __LINE__);
        }

        //
        // Get a local copy of the latest data in float format.
        //
        TMP006DataTemperatureGetFloat(&g_sTMP006Inst, &fAmbient, &fObject);

        //
        // Publish the data to the global structure for consumption by other
        // tasks.
        //
        xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
        g_sTMP006Data.fAmbient = fAmbient;
        g_sTMP006Data.fObject = fObject;
        g_sTMP006Data.xTimeStampTicks = xTaskGetTickCount();
        xSemaphoreGive(g_xCloudDataSemaphore);

    }

}

//*****************************************************************************
//
// Initializes the TMP006 task.
//
//*****************************************************************************
uint32_t
TMP006TaskInit(void)
{
    //
    // For BoosterPack 2 interface use PP5.
    //

    //
    // Reset the GPIO port to clear any previous interrupt configuration.
    //
    SysCtlPeripheralReset(SYSCTL_PERIPH_GPIOH);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOH))
    {
        //
        // Do Nothing. Wait for peripheral to complete reset.
        //
    }

    //
    // Configure and Enable the GPIO interrupt. Used for DRDY from the TMP006
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTH_BASE, GPIO_PIN_2);
    GPIOIntEnable(GPIO_PORTH_BASE, GPIO_PIN_2);
    ROM_GPIOIntTypeSet(GPIO_PORTH_BASE, GPIO_PIN_2, GPIO_FALLING_EDGE);

    //
    // Adjust interrupt priority so that this interrupt can safely call
    // FreeRTOS APIs.
    //
    IntPrioritySet(INT_GPIOH, 0xE0);
    ROM_IntEnable(INT_GPIOH);

    //
    // Create binary semaphores for flow control of the task synchronizing with
    // the I2C data read complete ISR (callback) and the GPIO pin ISR that
    // alerts software that the sensor has data ready to be read.
    //
    vSemaphoreCreateBinary(g_xTMP006DataReadySemaphore);
    vSemaphoreCreateBinary(g_xTMP006TransactionCompleteSemaphore);

    //
    // Create the switch task.
    //
    if(xTaskCreate(TMP006Task, (const portCHAR *)"TMP006    ",
                   TMP006_TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY +
                   PRIORITY_TMP006_TASK, g_xTMP006Handle) != pdTRUE)
    {
        //
        // Task creation failed.
        //
        return(1);
    }

    //
    // Check if Semaphore creation was successfull.
    //
    if((g_xTMP006DataReadySemaphore == NULL) ||
       (g_xTMP006TransactionCompleteSemaphore == NULL))
    {
        //
        // At least one semaphore was not created successfully.
        //
        return(1);
    }

    //
    // Set the active flag that shows this task is running properly.
    //
    g_sTMP006Data.bActive = true;
    g_sTMP006Data.fAmbient = 0.0f;
    g_sTMP006Data.fObject = 0.0f;
    g_sTMP006Data.xTimeStampTicks = 0;

    //
    // Initialize the cloud data struct to point to the local version of this
    // sensor data.
    //
    g_sCloudData.psTMP006Data = &g_sTMP006Data;

    //
    // Success.
    //
    return(0);
}
