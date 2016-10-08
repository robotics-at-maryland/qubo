//*****************************************************************************
//
// isl29023_task.c - Task that captures data from the light sensor over I2C.
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
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/sysctl.h"
#include "sensorlib/hw_isl29023.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/isl29023.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "priorities.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "isl29023_task.h"
#include "tmp006_task.h"
#include "bmp180_task.h"
#include "compdcm_task.h"
#include "sht21_task.h"
#include "command_task.h"
#include "cloud_task.h"
#include "drivers/pinout.h"
#include "drivers/buttons.h"

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
// Define ISL29023 I2C Address.
//
//*****************************************************************************
#define ISL29023_I2C_ADDRESS    0x44

//*****************************************************************************
//
// Global instance structure for the ISL29023 sensor driver.
//
//*****************************************************************************
tISL29023 g_sISL29023Inst;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the I2C transaction to
// complete. Sensor data has been read and is now ready for processing.
//
//*****************************************************************************
xSemaphoreHandle g_xISL29023TransactionCompleteSemaphore;

//*****************************************************************************
//
// Binary semaphore to sync between the ISL29023 threshold GPIO interrupt and
// the task.
//
//*****************************************************************************
xSemaphoreHandle g_xISL29023AdjustRangeSemaphore;

//*****************************************************************************
//
// Global new error flag to store the error condition if encountered.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8ISL29023I2CErrorStatus;

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
xTaskHandle g_xISL29023Handle;

//*****************************************************************************
//
// Global instance structure for the ISL29023 sensor data to be published.
//
//*****************************************************************************
sISL29023Data_t g_sISL29023Data;

//*****************************************************************************
//
// Constants to hold the floating point version of the thresholds for each
// range setting. Numbers represent an 81% and 19 % threshold levels. This
// creates a +/- 1% hysteresis band between range adjustments.
//
//*****************************************************************************
const float g_fThresholdHigh[4] =
{
    810.0f, 3240.0f, 12960.0f, 64000.0f
};

const float g_fThresholdLow[4] =
{
    0.0f, 760.0f, 3040.0f, 12160.0f
};

//*****************************************************************************
//
// ISL29023 Sensor callback function.  Called at the end of sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations and display. We
// also give back the I2C Semaphore so that other may use the I2C driver.
//
//*****************************************************************************
void
ISL29023AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
{
    portBASE_TYPE xHigherPriorityTaskWokenTransaction;
    xHigherPriorityTaskWokenTransaction = pdFALSE;

    //
    // Give the binary semaphore to wake the task so it can resume
    // and check the status error code and process data collected from the
    // sensor.
    //
    xSemaphoreGiveFromISR(g_xISL29023TransactionCompleteSemaphore,
                          &xHigherPriorityTaskWokenTransaction);

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8ISL29023I2CErrorStatus = ui8Status;

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
// ISL29023 Application error handler.
//
//*****************************************************************************
void
ISL29023AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1mISL29023 I2C Error: %d, File: %s, Line: %d\n",
               g_vui8ISL29023I2CErrorStatus, pcFilename, ui32Line);
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
    g_sISL29023Data.bActive = false;
    g_sISL29023Data.xTimeStampTicks = xTaskGetTickCount();
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
ISL29023DataEncodeJSON(char *pcBuf, uint32_t ui32BufSize)
{
    uint32_t ui32SpaceUsed;
    char pcVisibleBuf[12];
    char pcInfraredBuf[12];

    //
    // Convert the floating point members of the struct into strings.
    //
    uftostr(pcVisibleBuf, 12, 3, g_sISL29023Data.fVisible);
    uftostr(pcInfraredBuf, 12, 3, g_sISL29023Data.fInfrared);

    //
    // Merge the strings into the buffer in JSON format.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "{\"sISL29023Data_t\":{"
                              "\"bActive\":%d,\"fVisible\":%s,"
                              "\"fInfrared\":%s,\"ui8Range\":%d}}",
                              g_sISL29023Data.bActive, pcVisibleBuf,
                              pcInfraredBuf, g_sISL29023Data.ui8Range);

    //
    // Return size of string created.
    //
    return ui32SpaceUsed;

}

//*****************************************************************************
//
// ISL29023 Data print function.  Takes the float versions of sensor data and
// prints them to the VCP in a pretty format.
//
//*****************************************************************************
void ISL29023DataPrint(float fVisible)
{
    char pcVisibleBuf[25];

    //
    // Convert the floating point lux measurement to a string.
    //
    uftostr(pcVisibleBuf, 25, 3, fVisible);

    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print the light data as integer and fraction parts.
    //
    UARTprintf("ISL29023:\tVisible Lux:\t%s\tRange:\t\t%d\n", pcVisibleBuf,
               g_sISL29023Inst.ui8Range);

    //
    // Give back the UART Semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);
}

//*****************************************************************************
//
// Intensity and Range Tracking Function.  This adjusts the range and interrupt
// thresholds as needed.  Uses an 80/20 rule. If light is greather then 80% of
// maximum value in this range then go to next range up. If less than 20% of
// potential value in this range go the next range down.
//
//*****************************************************************************
void
ISL29023AppAdjustRange(float fVisible)
{
    uint8_t ui8NewRange;

    //
    // Initialize the local variable.
    //
    ui8NewRange = g_sISL29023Inst.ui8Range;

    //
    // Check if we crossed the upper threshold.
    //
    if(fVisible > g_fThresholdHigh[g_sISL29023Inst.ui8Range])
    {
        //
        // The current intensity is over our threshold so adjsut the range
        // accordingly
        //
        if(g_sISL29023Inst.ui8Range < ISL29023_CMD_II_RANGE_64K)
        {
            ui8NewRange = g_sISL29023Inst.ui8Range + 1;
        }
    }

    //
    // Check if we crossed the lower threshold
    //
    if(fVisible < g_fThresholdLow[g_sISL29023Inst.ui8Range])
    {
        //
        // If possible go to the next lower range setting and re-config the
        // thresholds.
        //
        if(g_sISL29023Inst.ui8Range > ISL29023_CMD_II_RANGE_1K)
        {
            ui8NewRange = g_sISL29023Inst.ui8Range - 1;
        }
    }

    //
    // If the desired range value changed then send the new range to the sensor
    //
    if(ui8NewRange != g_sISL29023Inst.ui8Range)
    {
        //
        // Take the I2C semaphore.
        //
        xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

        //
        // Perform a read, modify, write over I2C to set the new range.
        //
        ISL29023ReadModifyWrite(&g_sISL29023Inst, ISL29023_O_CMD_II,
                                ~ISL29023_CMD_II_RANGE_M, ui8NewRange,
                                ISL29023AppCallback, &g_sISL29023Inst);

        //
        // Wait for transaction to complete
        //
        xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, portMAX_DELAY);

        //
        // Give back the I2C Semaphore
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // If an error occurred call the error handler immediately.
        //
        if(g_vui8ISL29023I2CErrorStatus)
        {
            ISL29023AppErrorHandler(__FILE__, __LINE__);
        }
    }
}


//*****************************************************************************
//
// Called by the NVIC as a result of GPIO port E interrupt event. For this
// application GPIO port E pin 5 is the interrupt line for the ISL29023
//
// Notifies the application that light is outside of threshold limits.
// Task will poll the semaphore and adjust the ranges accordingly.
//
//*****************************************************************************
void
IntHandlerGPIOPortE(void)
{
    unsigned long ulStatus;
    portBASE_TYPE xHigherPriorityTaskWoken;

    ulStatus = GPIOIntStatus(GPIO_PORTE_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    GPIOIntClear(GPIO_PORTE_BASE, ulStatus);

    if(ulStatus & GPIO_PIN_5)
    {
        //
        // ISL29023 has indicated that the light level has crossed outside of
        // the intensity threshold levels set in INT_LT and INT_HT registers.
        //
        xSemaphoreGiveFromISR(g_xISL29023AdjustRangeSemaphore,
                              &xHigherPriorityTaskWoken);

        //
        // If the give of this semaphore causes a task to be ready then
        // make sure it has opportunity to run immediately upon return from
        // this ISR.
        //
        if(xHigherPriorityTaskWoken == pdTRUE)
        {
            portYIELD_FROM_ISR(true);
        }
    }
}


//*****************************************************************************
//
// This task captures data from the ISL29023 sensor and puts it into the shared
// data structure.
//
//*****************************************************************************
static void
ISL29023Task(void *pvParameters)
{
    portTickType xLastWakeTime;
    float fVisible;
    uint8_t ui8Mask;

    //
    // The binary semaphore is created full so we empty it first so we can
    // use it to wait for the AppCallback function.
    //
    xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, 0);
    xSemaphoreTake(g_xISL29023AdjustRangeSemaphore, 0);

    //
    // Take the I2C semaphore so we can init the sensor. Keep it until all init
    // is complete for this sensor.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Initialize the ISL29023 Driver.
    //
    ISL29023Init(&g_sISL29023Inst, &g_sI2CInst, ISL29023_I2C_ADDRESS,
                 ISL29023AppCallback, &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ISL29023I2CErrorStatus)
    {
        //
        // Give back the I2C Semaphore
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // Call the error handler.
        //
        ISL29023AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Configure the ISL29023 to measure Visible light continuously. Set a 8
    // sample persistence before the INT pin is asserted. Clears the INT flag.
    // Persistence setting of 8 is sufficient to ignore camera flashes.
    //
    ui8Mask = (ISL29023_CMD_I_OP_MODE_M | ISL29023_CMD_I_INT_PERSIST_M |
               ISL29023_CMD_I_INT_FLAG_M);
    ISL29023ReadModifyWrite(&g_sISL29023Inst, ISL29023_O_CMD_I, ~ui8Mask,
                            (ISL29023_CMD_I_OP_MODE_ALS_CONT |
                             ISL29023_CMD_I_INT_PERSIST_8),
                            ISL29023AppCallback, &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ISL29023I2CErrorStatus)
    {
        //
        // Give back the I2C Semaphore
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // Call the Error handler.
        //
        ISL29023AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Configure the upper threshold to 80% of maximum value
    //
    g_sISL29023Inst.pui8Data[1] = 0xCC;
    g_sISL29023Inst.pui8Data[2] = 0xCC;
    ISL29023Write(&g_sISL29023Inst, ISL29023_O_INT_HT_LSB,
                  g_sISL29023Inst.pui8Data, 2, ISL29023AppCallback,
                  &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ISL29023I2CErrorStatus)
    {
        //
        // Give back the I2C Semaphore
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // Call the error handler.
        //
        ISL29023AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Configure the lower threshold to 20% of maximum value
    //
    g_sISL29023Inst.pui8Data[1] = 0x33;
    g_sISL29023Inst.pui8Data[2] = 0x33;
    ISL29023Write(&g_sISL29023Inst, ISL29023_O_INT_LT_LSB,
                  g_sISL29023Inst.pui8Data, 2, ISL29023AppCallback,
                  &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ISL29023I2CErrorStatus)
    {
        ISL29023AppErrorHandler(__FILE__, __LINE__);
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
        vTaskDelayUntil(&xLastWakeTime, ISL29023_TASK_PERIOD_MS /
                                        portTICK_RATE_MS);

        //
        // Take the I2C semaphore.  Given back in the Interrupt contect in the
        // callback function after I2C transaction is complete.
        //
        xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

        //
        // Go get the latest data from the sensor.
        //
        ISL29023DataRead(&g_sISL29023Inst, ISL29023AppCallback,
                         &g_sISL29023Inst);

        //
        // Wait for the I2C Driver to tell us that transaction is complete.
        //
        xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore, portMAX_DELAY);

        //
        // Give back the I2C Semaphore so other can use the I2C interface.
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // If an error occurred call the error handler immediately.
        //
        if(g_vui8ISL29023I2CErrorStatus)
        {
            ISL29023AppErrorHandler(__FILE__, __LINE__);
        }

        //
        // Get a local floating point copy of the latest light data
        //
        ISL29023DataLightVisibleGetFloat(&g_sISL29023Inst, &fVisible);

        //
        // Check if the intensity of light has crossed a threshold. If so
        // then adjust range of sensor readings to track intensity.
        //
        if(xSemaphoreTake(g_xISL29023AdjustRangeSemaphore, 0) == pdTRUE)
        {
            //
            // Adjust the lux range.
            //
            ISL29023AppAdjustRange(fVisible);

            //
            // Take the I2C semaphore.  Given back in the Interrupt contect in
            // the callback function after I2C transaction is complete.
            //
            xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

            //
            // Now we must manually clear the flag in the ISL29023
            // register.
            //
            ISL29023Read(&g_sISL29023Inst, ISL29023_O_CMD_I,
                         g_sISL29023Inst.pui8Data, 1, ISL29023AppCallback,
                         &g_sISL29023Inst);

            //
            // Wait for the I2C Driver to tell us that transaction is complete.
            //
            xSemaphoreTake(g_xISL29023TransactionCompleteSemaphore,
                           portMAX_DELAY);

            //
            // Give back the I2C Semaphore so other can use the I2C interface.
            //
            xSemaphoreGive(g_xI2CSemaphore);

            //
            // If an error occurred call the error handler immediately.
            //
            if(g_vui8ISL29023I2CErrorStatus)
            {
                ISL29023AppErrorHandler(__FILE__, __LINE__);
            }
        }

        //
        // Publish the data to the global structure for consumption by other
        // tasks.
        //
        xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
        g_sISL29023Data.fVisible = fVisible;
        g_sISL29023Data.xTimeStampTicks = xTaskGetTickCount();
        g_sISL29023Data.ui8Range = g_sISL29023Inst.ui8Range;
        xSemaphoreGive(g_xCloudDataSemaphore);

    }
}

//*****************************************************************************
//
// Initializes the ISL29023 task.
//
//*****************************************************************************
uint32_t
ISL29023TaskInit(void)
{
    //
    // Reset the GPIO port to be sure all previous interrupt config is cleared.
    //
    SysCtlPeripheralReset(SYSCTL_PERIPH_GPIOE);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOE))
    {
        //
        // Do Nothing. Wait for reset to complete.
        //
    }

    //
    // Configure and Enable the GPIO interrupt. Used for INT signal from the
    // ISL29023
    //
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, GPIO_PIN_5);
    GPIOIntEnable(GPIO_PORTE_BASE, GPIO_PIN_5);
    ROM_GPIOIntTypeSet(GPIO_PORTE_BASE, GPIO_PIN_5, GPIO_FALLING_EDGE);

    //
    // Change the interrupt priority so that the interrupt handler function can
    // call FreeRTOS APIs.
    //
    IntPrioritySet(INT_GPIOE, 0xE0);
    ROM_IntEnable(INT_GPIOE);

    //
    // Create a transaction complete semaphore to sync the application callback
    // which is called from the I2C master driver in I2C interrupt context and
    // the task.
    //
    vSemaphoreCreateBinary(g_xISL29023TransactionCompleteSemaphore);

    //
    // Create a semaphore for notifying from the light threshold ISR to the task
    // that the light thresholds have been exceeded (too high or too low).
    //
    vSemaphoreCreateBinary(g_xISL29023AdjustRangeSemaphore);

    //
    // Create the switch task.
    //
    if(xTaskCreate(ISL29023Task, (const portCHAR *)"ISL29023  ",
                   ISL29023_TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY +
                   PRIORITY_ISL29023_TASK, g_xISL29023Handle) != pdTRUE)
    {
        //
        // Task creation failed.
        //
        return(1);
    }

    //
    // Check if Semaphore creation was successfull.
    //
    if((g_xISL29023TransactionCompleteSemaphore == NULL) ||
       (g_xISL29023AdjustRangeSemaphore == NULL))
    {
        //
        // Semaphore was not created successfully.
        //
        return(1);
    }

    //
    // Set the active flag that shows this task is running properly.
    // Initialize the other variables in the data structure.
    //
    g_sISL29023Data.bActive = true;
    g_sISL29023Data.fVisible = 0.0f;
    g_sISL29023Data.fInfrared = 0.0f;
    g_sISL29023Data.ui8Range = 0;
    g_sISL29023Data.xTimeStampTicks = 0;

    //
    // Initialize the cloud data global pointer to this sensors local data.
    //
    g_sCloudData.psISL29023Data = &g_sISL29023Data;

    //
    // Success.
    //
    return(0);

}
