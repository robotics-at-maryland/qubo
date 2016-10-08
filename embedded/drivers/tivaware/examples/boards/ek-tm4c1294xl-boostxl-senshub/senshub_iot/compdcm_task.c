//*****************************************************************************
//
// compdcm_task.c - Manage the 9-Axis sensor and Complimentary Filtered DCM.
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
#include "inc/hw_gpio.h"
#include "driverlib/debug.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/rom.h"
#include "sensorlib/hw_mpu9150.h"
#include "sensorlib/hw_ak8975.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/ak8975.h"
#include "sensorlib/mpu9150.h"
#include "sensorlib/comp_dcm.h"
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
// Define MPU9150 I2C Address.
//
//*****************************************************************************
#define MPU9150_I2C_ADDRESS     0x68

//*****************************************************************************
//
// Define how many iterations between a UART print update. This also
// currently defines how often the CompDCM data gets re-computed.
//
//*****************************************************************************
#define COMPDCM_PRINT_SKIP_COUNT 50

//*****************************************************************************
//
// Global instance structure for the ISL29023 sensor driver.
//
//*****************************************************************************
tMPU9150 g_sMPU9150Inst;

//*****************************************************************************
//
// Global Instance structure to manage the DCM state.
//
//*****************************************************************************
tCompDCM g_sCompDCMInst;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the I2C transaction to
// complete. Sensor data has been read and is now ready for processing.
//
//*****************************************************************************
xSemaphoreHandle g_xCompDCMTransactionCompleteSemaphore;

//*****************************************************************************
//
// Binary semaphore to control task flow and wait for the GPIO interrupt which
// indicates that the sensor has data that needs to be read.
//
//*****************************************************************************
xSemaphoreHandle g_xCompDCMDataReadySemaphore;

//*****************************************************************************
//
// Global new error flag to store the error condition if encountered.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8CompDCMI2CErrorStatus;

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
xTaskHandle g_xCompDCMHandle;

//*****************************************************************************
//
// Global instance structure for the BMP180 sensor data to be published.
//
//*****************************************************************************
sCompDCMData_t g_sCompDCMData;

//*****************************************************************************
//
// CompDCM Sensor callback function.  Called at the end of sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// give the Transaction complete semaphore so that task is triggered to do the
// bulk of data handling.
//
//*****************************************************************************
void
CompDCMAppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
{
    portBASE_TYPE xHigherPriorityTaskWokenTransaction;

    xHigherPriorityTaskWokenTransaction = pdFALSE;

    //
    // Let the task resume so that it can check the status flag and process
    // data.
    //
    xSemaphoreGiveFromISR(g_xCompDCMTransactionCompleteSemaphore,
                          &xHigherPriorityTaskWokenTransaction);

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8CompDCMI2CErrorStatus = ui8Status;

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
// CompDCM Application error handler.
//
//*****************************************************************************
void
CompDCMAppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1mCompDCM I2C Error: %d, File: %s, Line: %d\n",
               g_vui8CompDCMI2CErrorStatus, pcFilename, ui32Line);

    //
    // Tell users where to find more information about I2C errors and
    // reset the terminal color to normal.
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
    g_sCompDCMData.bActive = false;
    g_sCompDCMData.xTimeStampTicks = xTaskGetTickCount();
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
CompDCMDataEncodeJSON(char *pcBuf, uint32_t ui32BufSize)
{
    uint32_t ui32SpaceUsed, ui32Idx;
    char pcEulerBuf[3][12];
    char pcAccelerationBuf[3][12];
    char pcAngularVelocityBuf[3][12];
    char pcMagneticFieldBuf[3][12];
    char pcQuaternionBuf[4][12];

    //
    // Convert all floats in the structure to strings.
    //
    for(ui32Idx = 0; ui32Idx < 3; ui32Idx++)
    {
        uftostr(pcEulerBuf[ui32Idx], 12, 3,
                g_sCompDCMData.pfEuler[ui32Idx]);
        uftostr(pcAccelerationBuf[ui32Idx], 12, 3,
                g_sCompDCMData.pfAcceleration[ui32Idx]);
        uftostr(pcAngularVelocityBuf[ui32Idx], 12, 3,
                g_sCompDCMData.pfAngularVelocity[ui32Idx]);
        uftostr(pcMagneticFieldBuf[ui32Idx], 12, 3,
                g_sCompDCMData.pfMagneticField[ui32Idx]);
        uftostr(pcQuaternionBuf[ui32Idx], 12, 3,
                g_sCompDCMData.pfQuaternion[ui32Idx]);
    }

    //
    // Convert the last quaternion from float to string. Special handling
    // since we have four quaternions and three of everything else.
    //
    uftostr(pcQuaternionBuf[ui32Idx], 12, 3,
                 g_sCompDCMData.pfQuaternion[ui32Idx]);

    //
    // Merge all the strings together into a single JSON formated string.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "{\"sCompDCMData_t\":{"
                              "\"bActive\":%d,\"fEuler\":[%s,%s,%s],"
                              "\"fAcceleration\":[%s,%s,%s],"
                              "\"fAngularVelocity\":[%s,%s,%s],"
                              "\"fMagneticField\":[%s,%s,%s],"
                              "\"fQuaternion\":[%s,%s,%s,%s]}}",
                              g_sCompDCMData.bActive, pcEulerBuf[0],
                              pcEulerBuf[1], pcEulerBuf[2],
                              pcAccelerationBuf[0], pcAccelerationBuf[1],
                              pcAccelerationBuf[2], pcAngularVelocityBuf[0],
                              pcAngularVelocityBuf[1], pcAngularVelocityBuf[2],
                              pcMagneticFieldBuf[0], pcMagneticFieldBuf[1],
                              pcMagneticFieldBuf[2], pcQuaternionBuf[0],
                              pcQuaternionBuf[1], pcQuaternionBuf[2],
                              pcQuaternionBuf[3]);

    //
    // Return how much of the buffer was used.
    //
    return ui32SpaceUsed;
}

//*****************************************************************************
//
// CompDCM Data print function.  Takes the float versions of ambient and object
// temperature and prints them to the UART in a pretty format.
//
//*****************************************************************************
void CompDCMDataPrint(float *pfRPY, float *pfQuaternion)
{
    uint32_t ui32Idx;
    float pfEulerDegrees[3];
    char pcEulerBuf[3][12];
    char pcQuaternionBuf[4][12];

    //
    // Convert Eulers to degrees. 180/PI = 57.29...
    // Convert Yaw to 0 to 360 to approximate compass headings.
    //
    pfEulerDegrees[0] = pfRPY[0] * 57.295779513082320876798154814105f;
    pfEulerDegrees[1] = pfRPY[1] * 57.295779513082320876798154814105f;
    pfEulerDegrees[2] = pfRPY[2] * 57.295779513082320876798154814105f;
    if(pfEulerDegrees[2] < 0)
    {
        pfEulerDegrees[2] += 360.0f;
    }

    //
    // Convert floats in the structure to strings.
    //
    for(ui32Idx = 0; ui32Idx < 3; ui32Idx++)
    {
        uftostr(pcEulerBuf[ui32Idx], 12, 3, pfEulerDegrees[ui32Idx]);
        uftostr(pcQuaternionBuf[ui32Idx], 12, 3, pfQuaternion[ui32Idx]);
    }

    //
    // Convert the last quaternion from float to string. Special handling
    // since we have four quaternions and three of everything else.
    //
    uftostr(pcQuaternionBuf[ui32Idx], 12, 3, pfQuaternion[ui32Idx]);

    //
    // Attempt to grab the UART semaphore so we can send the error info to the
    // user locally.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print the Quaternions data.
    //
    UARTprintf("\nCompDCM:\tQ1: %s\tQ2: %s\tQ3: %s\tQ4: %s\n",
               pcQuaternionBuf[0], pcQuaternionBuf[1], pcQuaternionBuf[2],
               pcQuaternionBuf[3]);

    //
    // Print the Quaternions data.
    //
    UARTprintf("CompDCM:\tRoll: %s\tPitch: %s\tYaw: %s\n", pcEulerBuf[0],
               pcEulerBuf[1], pcEulerBuf[2]);

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);

    }

//*****************************************************************************
//
// For this application GPIO port M pin 3 is the interrupt line for the MPU9150
// Handle the data ready indication and alert our task if needed.
//
//*****************************************************************************
void
IntHandlerGPIOPortM(void)
{
    portBASE_TYPE xHigherPriorityTaskWoken;
    uint32_t ui32Status;

    //
    // Read the GPIO pin interrupt status register.
    //
    ui32Status = GPIOIntStatus(GPIO_PORTM_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    GPIOIntClear(GPIO_PORTM_BASE, ui32Status);

    //
    // Check if the data ready pin is the one that caused the interrupt.
    //
    if(ui32Status & GPIO_PIN_3)
    {
        //
        // Give the MPU9150 Data Ready Pin binary semaphore. This will
        // release the main CompDCM task so it can go get the data.
        //
        xHigherPriorityTaskWoken = pdFALSE;
        xSemaphoreGiveFromISR(g_xCompDCMDataReadySemaphore,
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
// This task gathers data from the MPU9150, calculates board orientation in
// Euler angles (roll, pitch and yaw) as well as Quaternions. It then makes
// this data available to the other tasks.
//
//*****************************************************************************
static void
CompDCMTask(void *pvParameters)
{
    float pfMag[3], pfAccel[3], pfGyro[3];
    float pfQuaternion[4], pfEuler[3];
    uint32_t ui32CompDCMStarted;
    uint32_t ui32Idx;

    //
    // The binary semaphore is created full so we take it up front and use it
    // later to sync between this task and the AppCallback function which is in
    // the I2C interrupt context. Likewise the GPIO port interrupt.
    //
    xSemaphoreTake(g_xCompDCMTransactionCompleteSemaphore, 0);
    xSemaphoreTake(g_xCompDCMDataReadySemaphore, 0);

    //
    // Take the I2C semaphore. Keep it until all init is complete for this
    // sensor.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Initialize the MPU9150 Driver.
    //
    MPU9150Init(&g_sMPU9150Inst, &g_sI2CInst, MPU9150_I2C_ADDRESS,
                CompDCMAppCallback, &g_sMPU9150Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xCompDCMTransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // Check for I2C Errors and responds
    //
    if(g_vui8CompDCMI2CErrorStatus)
    {
        CompDCMAppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Take the I2C semaphore.
    //
    xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

    //
    // Write application specific sensor configuration such as filter settings
    // and sensor range settings.
    //
    g_sMPU9150Inst.pui8Data[0] = MPU9150_CONFIG_DLPF_CFG_94_98;
    g_sMPU9150Inst.pui8Data[1] = MPU9150_GYRO_CONFIG_FS_SEL_250;
    g_sMPU9150Inst.pui8Data[2] = (MPU9150_ACCEL_CONFIG_ACCEL_HPF_5HZ |
                                  MPU9150_ACCEL_CONFIG_AFS_SEL_2G);
    MPU9150Write(&g_sMPU9150Inst, MPU9150_O_CONFIG, g_sMPU9150Inst.pui8Data, 3,
                 CompDCMAppCallback, &g_sMPU9150Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xCompDCMTransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Check for I2C Errors and responds
    //
    if(g_vui8CompDCMI2CErrorStatus)
    {
        //
        // Give back the I2C Semaphore so other can use the I2C interface.
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // Call the error handler.
        //
        CompDCMAppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Configure the data ready interrupt pin output of the MPU9150.
    //
    g_sMPU9150Inst.pui8Data[0] = MPU9150_INT_PIN_CFG_INT_LEVEL |
                                    MPU9150_INT_PIN_CFG_INT_RD_CLEAR |
                                    MPU9150_INT_PIN_CFG_LATCH_INT_EN;
    g_sMPU9150Inst.pui8Data[1] = MPU9150_INT_ENABLE_DATA_RDY_EN;
    MPU9150Write(&g_sMPU9150Inst, MPU9150_O_INT_PIN_CFG,
                 g_sMPU9150Inst.pui8Data, 2, CompDCMAppCallback,
                 &g_sMPU9150Inst);

    //
    // Wait for transaction to complete
    //
    xSemaphoreTake(g_xCompDCMTransactionCompleteSemaphore, portMAX_DELAY);

    //
    // Give back the I2C Semaphore so other can use the I2C interface.
    //
    xSemaphoreGive(g_xI2CSemaphore);

    //
    // Check for I2C Errors and responds
    //
    if(g_vui8CompDCMI2CErrorStatus)
    {
        CompDCMAppErrorHandler(__FILE__, __LINE__);
    }

    //
    // Initialize the DCM system. 50 hz sample rate.
    // accel weight = .2, gyro weight = .8, mag weight = .2
    //
    CompDCMInit(&g_sCompDCMInst, 1.0f / 50.0f, 0.2f, 0.6f, 0.2f);

    //
    // Clear the flag showing if we have "started" the DCM.  Starting
    // requires an initial valid data set.
    //
    ui32CompDCMStarted = 0;

    while(1)
    {
        //
        // wait for the GPIO interrupt that tells us MPU9150 has data.
        //
        xSemaphoreTake(g_xCompDCMDataReadySemaphore, portMAX_DELAY);

        //
        // Take the I2C semaphore.
        //
        xSemaphoreTake(g_xI2CSemaphore, portMAX_DELAY);

        //
        // Use I2C to go get data from the sensor into local memory.
        //
        MPU9150DataRead(&g_sMPU9150Inst, CompDCMAppCallback, &g_sMPU9150Inst);

        //
        // Wait for transaction to complete
        //
        xSemaphoreTake(g_xCompDCMTransactionCompleteSemaphore, portMAX_DELAY);

        //
        // Give back the I2C Semaphore so other can use the I2C interface.
        //
        xSemaphoreGive(g_xI2CSemaphore);

        //
        // Check for I2C Errors and responds
        //
        if(g_vui8CompDCMI2CErrorStatus)
        {
            CompDCMAppErrorHandler(__FILE__, __LINE__);
        }

        //
        // Get floating point version of the Accel Data in m/s^2.
        //
        MPU9150DataAccelGetFloat(&g_sMPU9150Inst, pfAccel, pfAccel + 1,
                                 pfAccel + 2);

        //
        // Get floating point version of angular velocities in rad/sec
        //
        MPU9150DataGyroGetFloat(&g_sMPU9150Inst, pfGyro, pfGyro + 1,
                                pfGyro + 2);

        //
        // Get floating point version of magnetic fields strength in tesla
        //
        MPU9150DataMagnetoGetFloat(&g_sMPU9150Inst, pfMag, pfMag + 1,
                                   pfMag + 2);

        //
        // Check if this is our first data ever.
        //
        if(ui32CompDCMStarted == 0)
        {
            //
            // Set flag indicating that DCM is started.
            // Perform the seeding of the DCM with the first data set.
            //
            ui32CompDCMStarted = 1;
            CompDCMMagnetoUpdate(&g_sCompDCMInst, pfMag[0], pfMag[1],
                                 pfMag[2]);
            CompDCMAccelUpdate(&g_sCompDCMInst, pfAccel[0], pfAccel[1],
                               pfAccel[2]);
            CompDCMGyroUpdate(&g_sCompDCMInst, pfGyro[0], pfGyro[1],
                              pfGyro[2]);
            CompDCMStart(&g_sCompDCMInst);
        }
        else
        {
            //
            // DCM Is already started.  Perform the incremental update.
            //
            CompDCMMagnetoUpdate(&g_sCompDCMInst, pfMag[0], pfMag[1],
                                 pfMag[2]);
            CompDCMAccelUpdate(&g_sCompDCMInst, pfAccel[0], pfAccel[1],
                               pfAccel[2]);
            CompDCMGyroUpdate(&g_sCompDCMInst, -pfGyro[0], -pfGyro[1],
                              -pfGyro[2]);
            CompDCMUpdate(&g_sCompDCMInst);

            //
            // Get Euler data. (Roll Pitch Yaw)
            //
            CompDCMComputeEulers(&g_sCompDCMInst, pfEuler, pfEuler + 1,
                                 pfEuler + 2);

            //
            // Get Quaternions.
            //
            CompDCMComputeQuaternion(&g_sCompDCMInst, pfQuaternion);

            //
            // Publish the data to the global structure for consumption by other
            // tasks.
            //
            xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
            for(ui32Idx = 0; ui32Idx < 3; ui32Idx++)
            {
                g_sCompDCMData.pfEuler[ui32Idx] = pfEuler[ui32Idx];
                g_sCompDCMData.pfQuaternion[ui32Idx] = pfQuaternion[ui32Idx];
                g_sCompDCMData.pfAngularVelocity[ui32Idx] = pfGyro[ui32Idx];
                g_sCompDCMData.pfMagneticField[ui32Idx] = pfMag[ui32Idx];
                g_sCompDCMData.pfAcceleration[ui32Idx] = pfAccel[ui32Idx];
            }

            //
            // Handle the 4th quaternion outside the loop.
            //
            g_sCompDCMData.pfQuaternion[3] = pfQuaternion[3];
            g_sCompDCMData.xTimeStampTicks = xTaskGetTickCount();

            //
            // Return the semaphore so others can modify the global data.
            //
            xSemaphoreGive(g_xCloudDataSemaphore);
        }
    }
}

//*****************************************************************************
//
// Initializes the CompDCM task.
//
//*****************************************************************************
uint32_t
CompDCMTaskInit(void)
{
    //
    // Reset the GPIO port M to make sure that all previous configuration is
    // cleared.
    //
    SysCtlPeripheralReset(SYSCTL_PERIPH_GPIOM);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOM))
    {
        //
        // Do nothing, waiting.
        //
    }

    //
    // Configure and Enable the GPIO interrupt. Used for DRDY from the MPU9150
    //
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTM_BASE, GPIO_PIN_3);
    GPIOIntEnable(GPIO_PORTM_BASE, GPIO_PIN_3);
    ROM_GPIOIntTypeSet(GPIO_PORTM_BASE, GPIO_PIN_3, GPIO_FALLING_EDGE);

    //
    // Must adjust the priority of the interrupt so that it can call FreeRTOS
    // APIs.
    //
    IntPrioritySet(INT_GPIOM, 0xE0);
    ROM_IntEnable(INT_GPIOM);

    //
    // Create binary semaphores for flow control of the task synchronizing with
    // the I2C data read complete ISR (callback) and the GPIO pin ISR that
    // alerts software that the sensor has data ready to be read.
    //
    vSemaphoreCreateBinary(g_xCompDCMDataReadySemaphore);
    vSemaphoreCreateBinary(g_xCompDCMTransactionCompleteSemaphore);

    //
    // Create the compdcm task itself.
    //
    if(xTaskCreate(CompDCMTask, (const portCHAR *)"CompDCM   ",
                   COMPDCM_TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY +
                   PRIORITY_COMPDCM_TASK, g_xCompDCMHandle) != pdTRUE)
    {
        //
        // Task creation failed.
        //
        return(1);
    }

    //
    // Check if Semaphore creation was successful.
    //
    if((g_xCompDCMDataReadySemaphore == NULL) ||
       (g_xCompDCMTransactionCompleteSemaphore == NULL))
    {
        //
        // At least one semaphore was not created successfully.
        //
        return(1);
    }

    //
    // Set the active flag that shows this task is running properly.
    //
    g_sCompDCMData.bActive = true;
    g_sCompDCMData.xTimeStampTicks = 0;

    //
    // point the global data structure to my local task data structuer.
    //
    g_sCloudData.psCompDCMData = &g_sCompDCMData;

    //
    // Success.
    //
    return(0);
}
