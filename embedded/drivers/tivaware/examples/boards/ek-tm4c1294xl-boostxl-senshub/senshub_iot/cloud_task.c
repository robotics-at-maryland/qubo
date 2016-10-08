//*****************************************************************************
//
// cloud_task.c - Task to connect and communicate to the cloud server.This
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

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "driverlib/interrupt.h"
#include "driverlib/timer.h"
#include "sensorlib/i2cm_drv.h"
#include "utils/cmdline.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "utils/lwiplib.h"
#include "utils/locator.h"
#include "drivers/pinout.h"
#include "drivers/buttons.h"
#include "drivers/eth_client_lwip.h"
#include "exosite.h"
#include "drivers/exosite_hal_lwip.h"
#include "third_party/exosite/exosite_meta.h"
#include "priorities.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "timers.h"
#include "isl29023_task.h"
#include "tmp006_task.h"
#include "bmp180_task.h"
#include "compdcm_task.h"
#include "sht21_task.h"
#include "command_task.h"
#include "cloud_task.h"

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
xTaskHandle g_xCloudTaskHandle;

//*****************************************************************************
//
// A global store for the current CIK.
//
//*****************************************************************************
char g_pcExositeCIK[50];

//*****************************************************************************
//
// A global storage location for the current proxy server URL.
//
//*****************************************************************************
char g_pcProxy[128];

//*****************************************************************************
//
// A global status flag to indicate if we are connected to exosite or not.
//
//*****************************************************************************
bool g_bOnline;

//*****************************************************************************
//
// A global pointer to identify the cloud update timer. Passed to the callback
// function as an argument.
//
//*****************************************************************************
void *g_pvCloudTimerID;

//*****************************************************************************
//
// A handle to use when controlling and configuring the timer with FreeRTOS
// APIs.
//
//*****************************************************************************
xTimerHandle g_xCloudTimerHandle;

//*****************************************************************************
//
// A global pointer to identify the switch debounce timer. Passed to the
// callback function as an argument.
//
//*****************************************************************************
void *g_pvSwitchesTimerID;

//*****************************************************************************
//
// A handle to use when controlling and configuring the timer with FreeRTOS
// APIs.
//
//*****************************************************************************
xTimerHandle g_xSwitchesTimerHandle;

//*****************************************************************************
//
// A global queue so tasks can send requests to this task.
//
//*****************************************************************************
xQueueHandle g_xCloudTaskRequestQueue;

//*****************************************************************************
//
// A global struct to hold the LED and switch states for this app.
//
//*****************************************************************************
sBoardData_t g_sCloudBoardData;

//*****************************************************************************
//
// A global struct to hold task statistics and define size of the array.
//
//*****************************************************************************
sTaskStatistics_t g_sTaskStatistics;

//*****************************************************************************
//
// Sets up the additional lwIP raw API services provided by the application.
//
//*****************************************************************************
void
SetupServices(void *pvArg)
{
    uint8_t pui8MAC[6];

    //
    // Setup the device locator service.
    //
    LocatorInit();
    lwIPLocalMACGet(pui8MAC);
    LocatorMACAddrSet(pui8MAC);

    LocatorAppTitleSet("EK-TM4C1294XL Senshub IOT Demo");
}

//*****************************************************************************
//
// Function converts a floating point value to a string.
//
// \param fValue is the value to be converted.
// \param pcStr is a pointer to a character buffer where the result will be
// stored.
// \param ui32Size is the size of the buffer where the result is stored.
// \param ui32Precision is the number of digits after the decimal point.
// Result will be truncated after this many digits.
//
// This function performs a brute force conversion of a float to a string.
// First checks for a negative value and adds the '-' char if needed. Then
// casts the float to an integer and uses existing usnprintf from
// utils/ustdlib.c to convert the integer part to the string. Then loops
// through the decimal portion multiplying by 10 and using the same integer
// conversion for each position of precision. Returns when ui32Size is reached
// or conversion is complete.
//
// \return the number of characters written to the buffer.
//
//*****************************************************************************
uint32_t
uftostr(char * pcStr, uint32_t ui32Size, uint32_t ui32Precision, float fValue)
{
    uint32_t ui32Integer;
    uint32_t ui32SpaceUsed;
    uint32_t ui32PrecisionCounter;

    //
    // Initialize local variable.
    //
    ui32SpaceUsed = 0;

    //
    // decrement size to account for room for a null character.
    //
    ui32Size -= 1;

    //
    // Account for negative values.
    //
    if(fValue < 0.0f)
    {
        if(ui32Size > 1)
        {
            pcStr[0] = '-';
            ui32SpaceUsed = 1;
        }
    }

    //
    // Initialize the loop conditions.
    //
    ui32PrecisionCounter = 0;
    ui32Integer = 0;

    //
    // Perform the conversion.
    //
    while((ui32PrecisionCounter <= ui32Precision) &&
          (ui32SpaceUsed < ui32Size))
    {
        //
        // Convert the new integer part.
        //
        ui32Integer = (uint32_t) fValue;

        //
        // Use usnprintf to convert the integer part to a string.
        //
        ui32SpaceUsed += usnprintf(&(pcStr[ui32SpaceUsed]),
                                   ui32Size - ui32SpaceUsed,
                                   "%d", ui32Integer);
        //
        // Subtract off the previous integer part.  Subtracts zero on first
        // time through loop.
        //
        fValue = fValue - ((float) ui32Integer);

        //
        // Multiply by 10 so next time through the most significant remaining
        // decimal will be beceome the integer part.
        //
        fValue *= 10;

        //
        // iF this is first time through the loop then add the decimal point
        // after the integer in the string. Also makes sure that there is room
        // for the decimal point.
        //
        if((ui32PrecisionCounter == 0) && (ui32SpaceUsed < ui32Size))
        {
             pcStr[ui32SpaceUsed] = '.';
             ui32SpaceUsed++;
        }

        //
        // Increment the precision counter to so we only convert as much as the
        // caller asked for.
        //
        ui32PrecisionCounter++;
    }

    //
    // Check if we quit because we ran out of buffer space.
    //
    if(ui32SpaceUsed >= ui32Size)
    {
        //
        // Since we decremented size at the beginning we should still have room
        // for the null char.
        //
        pcStr[ui32Size] = '\0';

        //
        // Return amount of space used plus the number of precision digits that
        // were not accommodated.
        //
        return (ui32SpaceUsed + (ui32Precision - ui32PrecisionCounter));
    }

    //
    // Terminate the string with null character.
    //
    pcStr[ui32SpaceUsed] = '\0';


    //
    // Return the amount of buffer space used. Not including null character.
    //
    return (ui32SpaceUsed);

}

//*****************************************************************************
//
// This function will decode the LED data and received from the Exosite server.
//
//*****************************************************************************
uint32_t
CloudBoardDataDecodeExositeHTTP(char *pcBuf, uint32_t ui32BufSize)
{
    const char *pcAlias;
    const char *pcAliasNext;
    char *pcValue;
    unsigned long ulValue;
    uint32_t ui32Index;

    //
    // Initialize the first alias pointer.
    //
    pcAliasNext = pcBuf;

    for(ui32Index = 0; ui32Index < 5; ui32Index++)
    {
        //
        // Update the current alias pointer to be equal to the prior "next"
        // pointer.
        //
        pcAlias = pcAliasNext;

        //
        // The value pointer is set to first char after the '='
        //
        pcValue = ustrstr(pcAlias, "=") + 1;

        //
        // make sure the "=" was found in the string.
        //
        if(pcValue != NULL)
        {
            //
            // convert the string to an unsigned long and capture a pointer to
            // the next alias. The Alias buffer must be one bigger than the
            // number of total aliases to accommodate a write to the N+1
            // alias position.
            //
            ulValue = ustrtoul(pcValue, &pcAliasNext, 10);

            //
            // Increment the Alias pointer to bypass the '&' character.
            //
            pcAliasNext += 1;
        }
        else
        {
            //
            // if pcValue is set to NULL we could not find a unsigned long
            // number in the string so break out of the for loop and return.
            //
            break;
        }

        if(ustrncmp(pcAlias,"ledd4=",6) == 0)
        {
            //
            // LEDD4 currently controlled by Ethernet hardware, no local
            // action taken for this variable.
            //
            g_sCloudBoardData.pui8LED[3] = ulValue;
            continue;
        }

        if(ustrncmp(pcAlias, "ledd3=", 6) == 0)
        {
            //
            // LEDD3 currently controlled by Ethernet hardware, no local
            // action taken for this variable.
            //
            g_sCloudBoardData.pui8LED[2] = ulValue;
            continue;
        }

        if(ustrncmp(pcAlias, "ledd2=", 6) == 0)
        {
            //
            // write the new value to LED bank.
            //
            LEDWrite(CLP_D2, (ulValue) ? CLP_D2 : 0);

            //
            // Store the new value in our local data structure.
            //
            g_sCloudBoardData.pui8LED[1] = ulValue;
            continue;
        }

        if(ustrncmp(pcAlias, "ledd1=", 6) == 0)
        {
            //
            // write the new value to LED bank.
            //
            LEDWrite(CLP_D1, (ulValue) ? CLP_D1 : 0);

            //
            // Store the new value in our local data structure.
            //
            g_sCloudBoardData.pui8LED[0] = ulValue;
            continue;
        }

        if(ustrncmp(pcAlias, "connlpds=", 9) == 0)
        {
            //
            // connlpds not yet implemented.  It is a sum total of all
            // Connected LaunchPads currently connected to the server.
            //
            continue;
        }
    }


    return 0;

}


//*****************************************************************************
//
// This function will encode the users switches into the exosite HTTP POST/GET
// format.  Stores the formated string into pcBuf.
//
//*****************************************************************************
uint32_t
CloudBoardDataEncodeExositeHTTP(char *pcBuf, uint32_t ui32BufSize)
{
    uint32_t ui32SpaceUsed;

    //
    // Encode the board data to Exosite format.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "usrsw1=%d&usrsw2=%d&"
                              "ledd1=%d&ledd2=%d",
                              g_sCloudBoardData.ui32SwitchPressCount[0],
                              g_sCloudBoardData.ui32SwitchPressCount[1],
                              g_sCloudBoardData.pui8LED[0],
                              g_sCloudBoardData.pui8LED[1]);

    //
    // Return the buffer size used as reported by usnprintf.
    //
    return ui32SpaceUsed;

}

//*****************************************************************************
//
// This function encodes the task usage data into the buffer provided in JSON
// format.
//
// Caller must take CloudDataSemaphore prior to calling this function.
//
//*****************************************************************************
uint32_t
CloudTaskStatsEncodeJSON(char* pcBuf, uint32_t ui32BufSize)
{
    xTaskStatusType *pxTaskStatus;
    uint32_t *pui32NumTasks;
    float *pfUsage;
    uint32_t ui32Index, ui32SpaceUsed;
    char pcUsageBuf[12];

    //
    // Initialize local pointers.
    //
    pxTaskStatus = g_sTaskStatistics.pxTaskStatus;
    pui32NumTasks = &(g_sTaskStatistics.ui32NumActiveTasks);
    pfUsage = g_sTaskStatistics.pfCPUUsage;

    //
    // Start with the JSON array designator.
    //
    ui32SpaceUsed = usnprintf(pcBuf, ui32BufSize, "[");

    //
    // Loop for adding elements to the array from the task state struct.
    //
    for(ui32Index = 0; ui32Index < *pui32NumTasks; ui32Index++)
    {
        //
        // Convert percent usage to a string.
        //
        uftostr(pcUsageBuf, 12, 3, pfUsage[ui32Index]);

        //
        // Merge the strings into a JSON formated string.
        //
        ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed,
                               ui32BufSize - ui32SpaceUsed,
                               "{\"xTaskStatusType\":{\"Task Name\":\"%s\","
                               "\"State\":%d,\"Current Priority\":%d,"
                               "\"Base Priority\":%d,"
                               "\"Task Ticks\":%d,\"%% Usage\":%s,"
                               "\"Stack Max\":%d}}",
                               pxTaskStatus[ui32Index].pcTaskName,
                               pxTaskStatus[ui32Index].eCurrentState,
                               pxTaskStatus[ui32Index].uxCurrentPriority,
                               pxTaskStatus[ui32Index].uxBasePriority,
                               pxTaskStatus[ui32Index].ulRunTimeCounter,
                               pcUsageBuf,
                               pxTaskStatus[ui32Index].usStackHighWaterMark);

        //
        // Check if this is not the last item written to the JSON array.
        //
        if(ui32Index != (*pui32NumTasks - 1))
        {
            //
            // Add a comma between elements of the JSON array. No comma after
            // the last element.
            //
            ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed,
                                       ui32BufSize - ui32SpaceUsed, ",");
        }
        else
        {
            //
            // Add the close array designator to the last item in the array.
            //
            ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed,
                                       ui32BufSize - ui32SpaceUsed, "]");
        }
    }

    return ui32SpaceUsed;
}

//*****************************************************************************
//
// This function updates the task usage data from FreeRTOS.
//
// Caller must take CloudDataSemaphore prior to calling this function.
//
//*****************************************************************************
void
CloudTaskStatsUpdate(void)
{
    uint32_t ui32Index;
    xTaskStatusType *pxTaskStatus;
    unsigned long *pulTotalRunTime;
    uint32_t *pui32NumTasks;
    float *pfUsage;

    //
    // Initialize local pointers.
    //
    pxTaskStatus = g_sTaskStatistics.pxTaskStatus;
    pulTotalRunTime = &(g_sTaskStatistics.ulTotalRunTime);
    pui32NumTasks = &(g_sTaskStatistics.ui32NumActiveTasks);
    pfUsage = g_sTaskStatistics.pfCPUUsage;

    //
    // Get the current task system statistics from FreeRTOS.
    //
    *pui32NumTasks = uxTaskGetSystemState(pxTaskStatus, TASK_STATUS_ARRAY_SIZE,
                                          pulTotalRunTime);

    //
    // Calculate the CPU usage as a percent of total run time for each task.
    //
    for(ui32Index = 0; ui32Index < *pui32NumTasks; ui32Index++)
    {
        pfUsage[ui32Index] = ((float) pxTaskStatus[ui32Index].ulRunTimeCounter)
                             / ((float) *pulTotalRunTime);
        pfUsage[ui32Index] *= 100.0f;
    }
}

//*****************************************************************************
//
// This function prints a list of RTOS task statistics for this board.
//
// Calling function must take the cloud data semaphore prior to calling this
// function.
//
//*****************************************************************************
void
CloudTaskStatsPrint(void)
{
#if configGENERATE_RUN_TIME_STATS
    uint32_t ui32Index;
    xTaskStatusType *pxTaskStatus;
    uint32_t *pui32NumTasks;
    float *pfUsage;
    char pcUsageBuf[12];

    //
    // Initialize local pointers.
    //
    pxTaskStatus = g_sTaskStatistics.pxTaskStatus;
    pui32NumTasks = &(g_sTaskStatistics.ui32NumActiveTasks);
    pfUsage = g_sTaskStatistics.pfCPUUsage;

    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print column headers.
    //
    UARTprintf("\nTask\t\tState\tCurPrio\tBasePrio\tTicks\t%%Usage\tStack\n");

    for(ui32Index = 0; ui32Index < *pui32NumTasks; ui32Index++)
    {

        uftostr(pcUsageBuf, 12, 1, pfUsage[ui32Index]);

        UARTprintf("%s", pxTaskStatus[ui32Index].pcTaskName);

        if(ustrlen((char *)pxTaskStatus[ui32Index].pcTaskName) < 10)
        {
            UARTprintf("\t");
        }
        UARTprintf("\t%5d\t%5d\t%5d\t   %10d\t",
                   pxTaskStatus[ui32Index].eCurrentState,
                   pxTaskStatus[ui32Index].uxCurrentPriority,
                   pxTaskStatus[ui32Index].uxBasePriority,
                   pxTaskStatus[ui32Index].ulRunTimeCounter);
        UARTprintf("%s%%\t%5d\n",
                   pcUsageBuf,
                   pxTaskStatus[ui32Index].usStackHighWaterMark);
    }

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);

#else
    //
    // Take the UART semaphore to guarantee the sequence of messages on VCP.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print a message to tell users how to enable this feature.
    //
    UARTprintf("configGENERATE_RUN_TIME_STATS is not defined in "
               "FreeRTOSConfig.h\n");

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore, portMAX_DELAY);

#endif // #if configGENERATE_RUN_TIME_STATS

}


//*****************************************************************************
//
// Prints a help message to the UART to help with troubleshooting Exosite
// connection issues.
//
//*****************************************************************************
void
PrintConnectionHelp(void)
{
    //
    // Get the UART control semaphore.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);

    //
    // Print help tips for how to get connected to exosite.
    // Caller is responsible to take and give UART semapohre.
    //
    UARTprintf("Troubleshooting Exosite Connection:\n\n");

    UARTprintf("    + Make sure you are connected to the internet.\n\n");

    UARTprintf("    + Make sure you have created an Exosite profile.\n\n");

    UARTprintf("    + Make sure you have a \"Connected Launchpad\" device\n");
    UARTprintf("      created in your Exosite profile.\n\n");

    UARTprintf("    + Make sure your that your board's MAC address is\n");
    UARTprintf("      correctly registered with your exosite profile.\n\n");

    UARTprintf("    + If you have a CIK, make sure it matches the CIK for\n");
    UARTprintf("      this device in your online profile with Exosite.\n\n");

    UARTprintf("    + If you have a proxy, make sure to configure it using\n");
    UARTprintf("      this terminal. Type 'setproxy help' to get started.\n");
    UARTprintf("      Once the proxy is set, type 'activate' to obtain a\n");
    UARTprintf("      new CIK, or 'connect' to connect to exosite using an\n");
    UARTprintf("      existing CIK.\n\n");

    UARTprintf("    + Make sure your device is available for provisioning.\n");
    UARTprintf("      If you are not sure that provisioning is enabled,\n");
    UARTprintf("      check the Read Me First documentation or the online\n");
    UARTprintf("      exosite portal for more information.\n\n");

    //
    // Give back the UART semaphore.
    //
    xSemaphoreGive(g_xUARTSemaphore);
}

//*****************************************************************************
//
// Set the proxy value at the Ethernet Client layer.
//
//*****************************************************************************
void CloudProxySet(char * pcProxy)
{
    char * pcPort;
    uint16_t ui16ProxyPort;

    if(ustrncmp(pcProxy, "off", 3) == 0)
    {
        EthClientProxySet(NULL, 0);
    }
    else
    {
        //
        // Find the location of the " " which delimits server from port.
        //
        pcPort = ustrstr(pcProxy, " ");

        if(pcPort)
        {
            //
            // Insert the null character to terminate the server part of the
            // string. Then increment pcPort to point to the port number
            // portion.
            //
            *pcPort = '\0';
            pcPort++;

            //
            // Convert the port string to a number.
            //
            ui16ProxyPort = ustrtoul(pcPort, NULL, 10);

            //
            // Copy the proxy server to a global variable.  The EthClient layer
            // does not copy the address but uses the pointer we pass.
            // Therefore, we must send a pointer to a global and not a pointer
            // to something on the stack (such as a local variable).
            //
            usprintf(g_pcProxy, "%s", pcProxy);

            //
            // Set the proxy in the ethernet client layer.
            //
            EthClientProxySet(g_pcProxy, ui16ProxyPort);
        }
    }

}

//*****************************************************************************
//
// Synchronize with Exosite.  Send out latest data and get the latest from
// them.  As a general rule we first read then write.
//
//*****************************************************************************
bool CloudSyncExosite(void)
{
    char pcBuf[2048];
    char pcReceiveBuf[256];
    uint32_t ui32SpaceUsed;
    uint32_t ui32ReceiveLength;

    //
    // Build the list of aliases to be read.
    //
    usnprintf(pcBuf, 1024, "ledd1&ledd2&ledd3&ledd4&connlpds");

    //
    // Read the aliases from the server.
    //
    ui32ReceiveLength = Exosite_Read(pcBuf, pcReceiveBuf, 256);

    //
    // Check that the receive was successful and data is in the buffer.
    //
    if(ui32ReceiveLength > 0)
    {
        //
        // Take the CloudData semaphore so the decode function can safely write
        // the values received from the cloud.
        //
        xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);

        //
        // Decode and do something with the received data.
        //
        CloudBoardDataDecodeExositeHTTP(pcReceiveBuf, ui32ReceiveLength);

        //
        // Give back the CloudData semaphore.
        //
        xSemaphoreGive(g_xCloudDataSemaphore);
    }

    //
    // Take the Cloud Data Semaphore so we have a current snapshot that won't
    // be changed while we copy it to the local buffer.
    //
    xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);


    //
    // Add the HTTP alias for each sensor's data and then the sensor JSON data
    // itself to our buffer.
    //
    ui32SpaceUsed = usnprintf(pcBuf, 1024, "tmp006_json=");
    ui32SpaceUsed += TMP006DataEncodeJSON(pcBuf + ui32SpaceUsed,
                                          1024 - ui32SpaceUsed);
    ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed, 1024 - ui32SpaceUsed,
                               "&sht21_json=");
    ui32SpaceUsed += SHT21DataEncodeJSON(pcBuf + ui32SpaceUsed,
                                         1024 - ui32SpaceUsed);
    ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed, 1024 - ui32SpaceUsed,
                               "&isl29023_json=");
    ui32SpaceUsed += ISL29023DataEncodeJSON(pcBuf + ui32SpaceUsed,
                                            1024 - ui32SpaceUsed);
    ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed, 1024 - ui32SpaceUsed,
                               "&compdcm_json=");
    ui32SpaceUsed += CompDCMDataEncodeJSON(pcBuf + ui32SpaceUsed,
                                           1024 - ui32SpaceUsed);
    ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed, 1024 - ui32SpaceUsed,
                               "&bmp180_json=");
    ui32SpaceUsed += BMP180DataEncodeJSON(pcBuf + ui32SpaceUsed,
                                         1024 - ui32SpaceUsed);

    //
    // For the Board Data variables they are directly in HTTP Exosite format
    // not in JSON.  So add just the '&' to show another alias is coming.
    // the alias is added to the buffer in the EncodeExositeHTTP function.
    //
    ui32SpaceUsed += usnprintf(pcBuf + ui32SpaceUsed, 1024 - ui32SpaceUsed,
                               "&");
    ui32SpaceUsed += CloudBoardDataEncodeExositeHTTP(pcBuf + ui32SpaceUsed,
                                                     1024 - ui32SpaceUsed);

    //
    // Cloud data copied to local buffer.  Give back the semaphore.
    //
    xSemaphoreGive(g_xCloudDataSemaphore);

    //
    // Write the buffer to the cloud server.
    //
    Exosite_Write(pcBuf, ui32SpaceUsed);

    //
    // Take the Cloud Data Semaphore so we have a current snapshot that won't
    // be changed while we copy it to the send buffer.
    //
    xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);

    //
    // The stats buffer is treated different because it is very large.  We will
    // send it as a seperate write transaction.  We add the exosite HTTP
    // formatting to the this buffer directly prior to writing the JSON encoded
    // task state data.
    //
    ui32SpaceUsed = usnprintf(pcBuf, 2048, "taskstats_json=");
    ui32SpaceUsed += CloudTaskStatsEncodeJSON(pcBuf + ui32SpaceUsed,
                                              2048 - ui32SpaceUsed);

    //
    // Cloud data copied to local buffer.  Give back the semaphore.
    //
    xSemaphoreGive(g_xCloudDataSemaphore);

    //
    // Write the Task State data now.
    //
    Exosite_Write(pcBuf, ui32SpaceUsed);

    //
    // Return.
    //
    return true;
}

//*****************************************************************************
//
// Connect to exosite and attempt to register as a new device.  The device
// MAC address must already be entered into the exosite system by the user at
// http://ti.exosite.com
//
//*****************************************************************************
bool
CloudProvisionExosite(void)
{
    uint32_t ui32Idx;

    //
    // If we get here, no CIK was found in EEPROM storage. We may need to
    // obtain a CIK from the server.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
    UARTprintf("Connecting to exosite to obtain a new CIK...\n");
    xSemaphoreGive(g_xUARTSemaphore);

    //
    // Try to activate with Exosite a few times. If we succeed move on with the
    // new CIK. Otherwise, fail.
    //
    for(ui32Idx = 0; ui32Idx < 3; ui32Idx++)
    {
        if(Exosite_Activate())
        {
            //
            // If exosite gives us a CIK, send feedback to the user.
            //
            xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
            UARTprintf("CIK acquired!\n");
            xSemaphoreGive(g_xUARTSemaphore);

            if(Exosite_GetCIK(g_pcExositeCIK))
            {
              xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
              UARTprintf("CIK: %s\n", g_pcExositeCIK);
              xSemaphoreGive(g_xUARTSemaphore);
            }
            else
            {
              //
              // This shouldn't ever happen, but print an error message in
              // case it does.
              //
              xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
              UARTprintf("ERROR reading new CIK from EEPROM.\n");
              xSemaphoreGive(g_xUARTSemaphore);
            }

            //
            // Return "true" indicating that we found a valid CIK.
            //
            return true;
        }
        else
        {
            //
            xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
            UARTprintf("Attempt %d of %d to acquire CIK failed!\n",
                       ui32Idx + 1, 3);
            xSemaphoreGive(g_xUARTSemaphore);

            //
            // If the activation fails, wait at least one second before
            // retrying.
            //
            vTaskDelay(1000 / portTICK_RATE_MS);
        }
    }

    //
    // Exosite didn't respond, so let the user know.
    //
    xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
    UARTprintf("No CIK could be obtained.\n\n");
    xSemaphoreGive(g_xUARTSemaphore);

    //
    // Print a longer troubleshooting message to help the user figure out why
    // a CIK was not obtained.
    //
    PrintConnectionHelp();

    //
    // Return "false", indicating that no CIK was found.
    //
    return false;
}

//*****************************************************************************
//
// This function will start a new exosite session. Typically called at startup.
// Will first attempt to read the current CIK from EEPROM if successful will
// attempt a sync using that CIK.  If no CIK is found in EEPROM or the sync
// with the EEPROM CIK fails then automatically attempt to provision with
// Exosite to obtain a new CIK.
//
//*****************************************************************************
void
CloudStartExosite(void)
{
    uint32_t ui32Index;

    //
    // Check the EEPROM for a valid CIK first.
    //
    if(Exosite_GetCIK(g_pcExositeCIK))
    {
        //
        // EEPROM CIK Found.
        //
        xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
        UARTprintf("CIK found in EEPROM: %s\n", g_pcExositeCIK);
        UARTprintf("Attempting initial connection...\n");
        xSemaphoreGive(g_xUARTSemaphore);

        //
        // Try several times to get a successful synchronization with the
        // Exosite server.
        //
        for(ui32Index = 0; ui32Index < 5; ui32Index++)
        {
            //
            // Do sync now to see if the CIK is still valid.
            //
            if(CloudSyncExosite())
            {
                //
                // Let user know we succeeeded and set Online state variable.
                xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
                UARTprintf("Connected!\n");
                xSemaphoreGive(g_xUARTSemaphore);
                g_bOnline = true;

                break;
            }
        }

        //
        // Check to see if we hit the maximum number of retry attempts.
        //
        if(ui32Index == 5)
        {
            //
            // Let user know that initial sync failed.
            //
            xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
            UARTprintf("Initial sync failed. CIK may be invalid.\n");
            xSemaphoreGive(g_xUARTSemaphore);

            //
            // Now try to get a new CIK from Exosite.
            //
            if(CloudProvisionExosite())
            {
                //
                // Success.
                //
                g_bOnline = true;
            }
            else
            {
                //
                // If both cases above fail, alert the user, but continue on
                // with the application.
                //
                g_bOnline = false;
            }
        }
    }
    else
    {
        //
        // EEPROM CIK NOT Found.
        //
        xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
        UARTprintf("CIK not found in EEPROM.\n");
        xSemaphoreGive(g_xUARTSemaphore);

        //
        // Attempt to get a new CIK from the server.
        //
        if(CloudProvisionExosite())
        {
            //
            // Success.
            //
            g_bOnline = true;
        }
        else
        {
            //
            // If both cases above fail, alert the user, but continue on
            // with the application.
            //
            g_bOnline = false;
        }
    }
}

//*****************************************************************************
//
// Callback function for the FreeRTOS software timer that triggers cloud
// updates.
//
//*****************************************************************************
void CloudTimerCallback(xTimerHandle xTimer)
{
    sCloudTaskRequest_t sRequest;

    //
    // Send a request to the queue to perform a sync to the cloud.
    // For the sync request the buffer is not used. This will wake the main
    // task function.
    //
    sRequest.ui32Request = CLOUD_REQUEST_SYNC;
    usprintf(sRequest.pcBuf, "SYNC REQUEST");
    xQueueSendToBack(g_xCloudTaskRequestQueue, (void *) &sRequest,
                     portMAX_DELAY);
}

//*****************************************************************************
//
// Callback function for the FreeRTOS software timer that polls and debounces
// users switches.
//
//*****************************************************************************
void SwitchTimerCallback(xTimerHandle xTimer)
{
    uint8_t ui8Buttons, ui8ButtonsChanged;

    //
    // Check the current debounced state of the buttons.
    //
    ui8Buttons = ButtonsPoll(&ui8ButtonsChanged,0);

    //
    // Take the cloud data semaphore
    //
    xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);

    //
    // If either button has been pressed, record that status to the
    // corresponding global variable.
    //
    if(BUTTON_PRESSED(USR_SW1, ui8Buttons, ui8ButtonsChanged))
    {
        g_sCloudBoardData.ui32SwitchPressCount[0] += 1;
    }
    else if(BUTTON_PRESSED(USR_SW2, ui8Buttons, ui8ButtonsChanged))
    {
        g_sCloudBoardData.ui32SwitchPressCount[1] += 1;
    }

    //
    // Give back the cloud data semaphore.
    //
    xSemaphoreGive(g_xCloudDataSemaphore);

}

//*****************************************************************************
//
// This function is the main task function that runs the cloud interface
// for this app. It also manages the LEDs and Buttons for the board.
//
//*****************************************************************************
static void
CloudTask(void *pvParameters)
{
    uint32_t ui32IPAddrOld, ui32IPAddrCurrent;
    uint8_t *pui8IPAddr;
    uint32_t ui32Timeout;
    sCloudTaskRequest_t sCloudRequest;

    //
    // Start the button polling timer.
    //
    xTimerStart(g_xSwitchesTimerHandle, portMAX_DELAY);

    //
    // Initialize the pointer for easier printing of the IP address.
    //
    pui8IPAddr = (uint8_t *) &ui32IPAddrCurrent;

    //
    // Loop for a few seconds while polling to see if we have a IP address yet.
    //
    for(ui32Timeout = 0; ui32Timeout < 20; ui32Timeout++)
    {
        //
        // Get the initial IP Address (likely all 'ff');
        //
        ui32IPAddrCurrent = lwIPLocalIPAddrGet();
        ui32IPAddrOld = ui32IPAddrCurrent;

        if((ui32IPAddrCurrent != 0xFFFFFFFF) && (ui32IPAddrCurrent != 0))
        {

            //
            // Now that we have a presumably valid IP print it for the user
            // to reference later.
            //
            xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
            UARTprintf("IP: %d.%d.%d.%d\n", pui8IPAddr[0], pui8IPAddr[1],
                       pui8IPAddr[2], pui8IPAddr[3]);
            xSemaphoreGive(g_xUARTSemaphore);

            //
            // Start the cloud connection.  First checks for existing saved
            // connection credentials.  If found will attempt a sync to prove
            // the credentials are still valid.  If not found or no longer
            // valid then it will try to establish a first time connection and
            // obtain a CIK.
            //
            CloudStartExosite();

            //
            // exit the polling loop early since we have an IP address and
            // have tried to connect to exosite.
            //
            break;
        }

        //
        // Delay for 500 milliseconds before checking again if we have an IP.
        //
        vTaskDelay(500 / portTICK_RATE_MS);
    }

    //
    // Check if we successfully connected to the cloud
    //
    if(g_bOnline)
    {
        //
        // Print the status update message.
        //
        xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
        UARTprintf("Connection to Exosite Successful.\n");
        UARTprintf("Cloud Task: CIK = %s", g_pcExositeCIK);
        xSemaphoreGive(g_xUARTSemaphore);
    }

    //
    // Start the cloud timer.
    //
    xTimerStart(g_xCloudTimerHandle, portMAX_DELAY);

    while(1)
    {
        //
        // Block forever until a message is put into the queue.
        //
        xQueueReceive(g_xCloudTaskRequestQueue, (void *) &sCloudRequest,
                      portMAX_DELAY);

        //
        // The first part of the queue message is the type of message, switch
        // on the type of message and perform actions accordingly.
        //
        switch(sCloudRequest.ui32Request)
        {
            case CLOUD_REQUEST_SYNC:
            {
                //
                // Update the FreeRTOS task statistics.
                //
                xSemaphoreTake(g_xCloudDataSemaphore, portMAX_DELAY);
                CloudTaskStatsUpdate();
                xSemaphoreGive(g_xCloudDataSemaphore);

                //
                // Sync data with the cloud server. Syncs all data not just
                // the task data.
                //
                CloudSyncExosite();
                break;
            }

            case CLOUD_REQUEST_ACTIVATE:
            {
                //
                // Call the provision function which attempts to acquire a new
                // CIK.
                //
                CloudProvisionExosite();
                break;
            }

            case CLOUD_REQUEST_START:
            {
                //
                // Call the start function which will connect and sync. Will
                // also acquire a new CIK if needed.
                //
                CloudStartExosite();
                break;
            }

            case CLOUD_REQUEST_PROXY_SET:
            {
                //
                // Set or clear the proxy address. As directed by user command
                // line.
                //
                CloudProxySet(sCloudRequest.pcBuf);
                break;
            }

            default:
            {
                //
                // Any other request type is not supported, alter the user.
                //
                xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
                UARTprintf("Cloud task request %d not supported.\n",
                           sCloudRequest.ui32Request);
                xSemaphoreGive(g_xUARTSemaphore);
                break;
            }
        }

        //
        // Check the latest IP address.
        //
        ui32IPAddrCurrent = lwIPLocalIPAddrGet();

        //
        // if the IP address changed then print that too.
        //
        if(ui32IPAddrCurrent != ui32IPAddrOld)
        {
            xSemaphoreTake(g_xUARTSemaphore, portMAX_DELAY);
            UARTprintf("IP: %d.%d.%d.%d\n", pui8IPAddr[0], pui8IPAddr[1],
                       pui8IPAddr[2], pui8IPAddr[3]);
            xSemaphoreGive(g_xUARTSemaphore);

            //
            // Set the old address variable equal to current so we only
            // display the address when it changes.
            //
            ui32IPAddrOld = ui32IPAddrCurrent;
        }
    }
}
//*****************************************************************************
//
// Initialize the CloudTask which manages communication with the cloud server.
//
//*****************************************************************************
uint32_t CloudTaskInit(void)
{
    uint32_t ui32User0, ui32User1;
    uint8_t pui8MAC[6];
    uint8_t ui8Idx;

    //
    // Init the pointer in the cloud structure to point to the local system
    // data structure.
    //
    g_sCloudData.psBoardData = &g_sCloudBoardData;

    //
    // Init the pointer in the cloud structure to point to the Task stats
    // data structure.
    //
    g_sCloudData.psTaskStatistics = &g_sTaskStatistics;

    //
    // Get the MAC address from the user registers.
    //
    ROM_FlashUserGet(&ui32User0, &ui32User1);
    if((ui32User0 == 0xffffffff) || (ui32User1 == 0xffffffff))
    {
        //
        // Tell the user why we failed.
        //
        UARTprintf("MAC Address Not Found!\n");

        //
        // Return Error since we cannot go online without a MAC address.
        //
        return(1);
    }
    else
    {
        //
        // Convert the 24/24 split MAC address from NV ram into a 32/16 split MAC
        // address needed to program the hardware registers, then program the MAC
        // address into the Ethernet Controller registers.
        //
        pui8MAC[0] = ((ui32User0 >>  0) & 0xff);
        pui8MAC[1] = ((ui32User0 >>  8) & 0xff);
        pui8MAC[2] = ((ui32User0 >> 16) & 0xff);
        pui8MAC[3] = ((ui32User1 >>  0) & 0xff);
        pui8MAC[4] = ((ui32User1 >>  8) & 0xff);
        pui8MAC[5] = ((ui32User1 >> 16) & 0xff);

        //
        // Tell users the current MAC address, needed for Exosite provisioning.
        //
        UARTprintf("Current MAC: ");

        //
        // Extract each pair of characters and print them to the UART.
        //
        for(ui8Idx = 0; ui8Idx < 5; ui8Idx++)
        {
            UARTprintf("%02x:", pui8MAC[ui8Idx]);
        }

        //
        // print the last part of the MAC.
        //
        UARTprintf("%02x\n", pui8MAC[ui8Idx]);
    }

    //
    // Lower the priority of the Ethernet interrupt handler.  This is required
    // so that the interrupt handler can safely call the interrupt-safe
    // FreeRTOS functions (specifically to send messages to the queue).
    //
    ROM_IntPrioritySet(INT_EMAC0, 0xE0);

    //
    // Initialize the lwIP stack and the exosite layers.
    //
    Exosite_Init("texasinstruments", "ek-tm4c1294xl", IF_ENET, 0);

    //
    // Setup the remaining services inside the TCP/IP thread's context.
    //
    tcpip_callback(SetupServices, 0);

    //
    // Create the queue used by other tasks to send requests to this task
    //
    g_xCloudTaskRequestQueue = xQueueCreate(CLOUD_QUEUE_LENGTH,
                                            CLOUD_QUEUE_ITEM_SIZE);

    //
    // Create the software timer to go off periodically and trigger updates to
    // and from the cloud server.  Triggers at the rate specified by
    // CLOUD_TIMER_PERIOD_MS, auto reloads for continuous operation,
    //
    g_xCloudTimerHandle = xTimerCreate((const char *)"Cloud Timer",
                                       CLOUD_TIMER_PERIOD_MS /
                                       portTICK_RATE_MS, pdTRUE,
                                       g_pvCloudTimerID, CloudTimerCallback);

    //
    // Create a software timer to go off periodically and update and debounce
    // the user switches (buttons).
    //
    g_xSwitchesTimerHandle = xTimerCreate((const char *) "Switch Timer",
                                          SWITCH_TIMER_PERIOD_MS /
                                          portTICK_RATE_MS, pdTRUE,
                                          g_pvSwitchesTimerID,
                                          SwitchTimerCallback);


    //
    // Create the cloud task.
    //
    if(xTaskCreate(CloudTask, (const portCHAR *)"Cloud",
                   CLOUD_TASK_STACK_SIZE, NULL, tskIDLE_PRIORITY +
                   PRIORITY_CLOUD_TASK, g_xCloudTaskHandle) != pdTRUE)
    {
        //
        // Task creation failed.
        //
        return(1);
    }

    if((g_xCloudTimerHandle != NULL) && (g_xCloudTaskRequestQueue != NULL) &&
       (g_xSwitchesTimerHandle != NULL))
    {
        //
        // Success.
        //
        return(0);
    }

    //
    // Either queue or timer create failed.
    //
    return 1;

}
