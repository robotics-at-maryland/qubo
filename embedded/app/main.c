/*
Copyright 2014, Jernej Kovacic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file
 *
 * A simple demo FreeRTOS application that receives characters via
 * UART 1 and prints the inverted text when 'Enter' is pressed.
 *
 * In parallel to this, a simple light show runs that periodically
 * turns on and off various combinations of built-in LEDS. The
 * light show may be paused or resumed by pressing the switch 1.
 *
 * Additionally it runs a few other tasks that regularly display
 * the system's uptime and reload the selcted watchdog to prevent
 * resets.
 *
 * @author Jernej Kovacic
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>

#include "FreeRTOSConfig.h"
#include "app_defaults.h"
#include "wdtask.h"
#include "print.h"
#include "receive.h"
#include "lightshow.h"

#include "uart.h"
#include "gpio.h"

#define DEF_TIMER_DELAY_SEC            ( 5 )
#define DEF_NR_TIMER_STR_BUFFERS       ( 3 )
#define DEF_RECV_QUEUE_SIZE            ( 3 )

/* Declaration of a function, implemented in nostdlib.c */
extern char* itoa(int32_t value, char* str, uint8_t radix);


/* Struct with settings for the periodic timer task */
typedef struct _timerParam_t
{
    uint32_t delay; /* delay in seconds */
} timerParam_t;


/*
 * Parameters for all tasks:
 */

/* Parameter for the wdtask(): */
static const wdDelayParam wdtaskPar =
    (wdDelayParam) { .delayMs = 8000 };

/* Parameter for the debug printGateKeeperTask: */
static printUartParam printDebugParam =
    (printUartParam) { .uartNr = APP_DEBUG_UART };

/* Parameter for the application's gate keeper task: */
static printUartParam printParam;

/* Parameter for the application's receiver task */
static recvUartParam recvParam;

/* Parameter for the periodic timer task */
static timerParam_t timerParam =
    (timerParam_t) { .delay = 1 };

/* Parameter for the switch 1 handling task */
static Switch1TaskParam_t sw1Param;

/* Parameter for the light show task */
static LightShowParam_t lsParam =
    (LightShowParam_t) { .delayMs = APP_TIMER_DELAY_SEC };

/* Fixed frequency periodic task function that displays system's uptime */
void vPeriodicTimerFunction(void* pvParameters)
{
	/* Buffers for strings to be printed: */
    portCHAR prstr[ 10 ][ DEF_NR_TIMER_STR_BUFFERS ];
    portCHAR buf[ 10 ];
    uint8_t cntr;

    const timerParam_t* const params = (timerParam_t*) pvParameters;
    const uint32_t delay = ( NULL!=params ? params->delay : APP_TIMER_DELAY_SEC );

    TickType_t lastWakeTime;
    uint32_t sec;
    uint32_t min;

    /* Initialization of variables */
    cntr = 0;
    sec = 0;
    min = 0;

    /*
     * This variable must be initialized once.
     * Then it will be updated automatically by vTaskDelayUntil().
     */
    lastWakeTime = xTaskGetTickCount();

    for( ; ; )
    {
        /* Appropriately update counters of seconds and minutes */
        min += (sec / 60);
        sec %= 60;

        /* Write number of minutes to print buffer string, followed by ':' */
        itoa(min, buf, 10);
        strcpy(prstr[cntr], buf);
        strcat(prstr[cntr], " : ");

        /* Followed by the number of seconds, always written with 2 digits */
        itoa(sec, buf, 10);
        if ( sec<10 )
        {
            strcat(prstr[cntr], "0");
        }
        strcat(prstr[cntr], buf);
        strcat(prstr[cntr], "\r\n");

        /* Print the system's uptime */
        vPrintMsg(APP_DEBUG_UART, prstr[cntr]);

        /* And switch to the next print buffer */
        ++cntr;
        cntr %= DEF_NR_TIMER_STR_BUFFERS;

        /*
         * The task will unblock exactly after 'delay' seconds (actually
         * after the appropriate number of ticks), relative from the moment
         * it was last unblocked.
         */
        vTaskDelayUntil( &lastWakeTime, (delay / portTICK_RATE_MS) * 1000 );

        /* Update the counter of seconds */
        sec += delay;
    }

    /*
     * If the task implementation ever manages to break out of the
     * infinite loop above, it must be deleted before reaching the
     * end of the function!
     */
    vTaskDelete(NULL);
}


/*
 * A task that receives a string from the receiver task, inverts
 * the string and sends it to the gate keeper task.
 *
 * @param params - ignored
 */
static void vInvertText(void* params)
{
    /* Handles to queues and related tasks: */
    QueueHandle_t strQueue = NULL;
    TaskHandle_t recvHandle = NULL;
    TaskHandle_t printHandle = NULL;

    /* Auxiliary variables for manipulation of strings: */
    portCHAR* str;
    portCHAR* p;
    portCHAR* s;
    portCHAR q;

	/* Initialize the UART 1 if necessary: */
    if ( 0 != APP_PRINT_UART_NR ||
         0 != APP_RECV_UART_NR )
    {
        /* UART 0 has been initialized before. */
        uart_config(
            APP_PRINT_UART_NR,
            DEF_UART1_PORT,
            DEF_UART1_PIN_RX,
            DEF_UART1_PIN_TX,
            DEF_UART1_PCTL,
            DEF_UART1_BR,
            DEF_UART1_DATA_BITS,
            DEF_UART1_PARITY,
            DEF_UART1_STOP
        );

        uart_enableUart(APP_PRINT_UART_NR);
    }

    /* Prepare both related tasks' param structs: */
    printParam.uartNr = APP_PRINT_UART_NR;

    recvParam.uartNr = APP_RECV_UART_NR;
    recvParam.queue = &strQueue;

    /*
     * Initialize a queue for communication between receiver and this task.
     * Note that at this situation it is assumed that the queue will
     * not be deleted while the receiver task is still operating, so it
     * is safe to allocate it in the task's stack. If this is  not the
     * case or if heap space is a scarce resource, consider allocation
     * of the queue as a global variable.
     */
    strQueue = xQueueCreate(DEF_RECV_QUEUE_SIZE, sizeof(portCHAR*));

    /*
     * Init and create receiver task. Should any operation fail,
     * skip to the cleanup part and finish the task.
     */
    if ( NULL == strQueue ||
         pdFAIL == printInit(APP_PRINT_UART_NR) ||
         pdFAIL == recvInit(APP_PRINT_UART_NR) ||
         pdPASS != xTaskCreate(recvTask, "procrecv", 128,
                 (void*) &recvParam, APP_PRIOR_RECEIVER, &recvHandle) )
    {
        goto cleanupProcTask;
    }

    /*
     * Note that the gate keeper task for the
     * UART 0 has already been created before.
     */
    if ( 0 != APP_PRINT_UART_NR &&
         pdPASS != xTaskCreate(printGateKeeperTask, "procgk", 128,
               (void*) &printParam, APP_PRIOR_PRINT_GATEKEEPER, &printHandle) )
    {
        goto cleanupProcTask;
    }


    for ( ; ; )
    {
        /* Wait until a pointer to a string is received from the receiver:*/
        xQueueReceive(strQueue, (void*) &str, portMAX_DELAY);

        /*
         * Invert the string by using two pointers,
         * one incrementing from the start and the other one
         * decrementing from the end of the string
         * ('\0' terminator not included!).
         */
        p = str;
        for ( s=str; '\0'!=*s; ++s );
        for ( --s; p < s; ++p, --s )
        {
             q = *p;
            *p = *s;
            *s =  q;
        }

        /* Append EOL to the processed string and print it: */
        strcat( str, "\r\n" );
        vPrintMsg(APP_PRINT_UART_NR, str);
    }


cleanupProcTask:

    /*
     * Cleanup section, should only be reached if initialization or
     * creation of related tasks failed. If this occurs,
     * check all handles and delete their corresponding synchronization
     * primitives and tasks.
     */

    if ( NULL != strQueue )
    {
        vQueueDelete(strQueue);
    }

    if ( NULL != recvHandle )
    {
        vTaskDelete(recvHandle);
    }

    if ( NULL!= printHandle )
    {
        vTaskDelete(printHandle);
    }

    /* Finally delete the task as well */
    vTaskDelete(NULL);

    /* Just to suppress a warning due to an ignored parameter */
    (void) params;
}


/*
 * A convenience function that is called when a FreeRTOS API call fails
 * and a program cannot continue. It prints a message (if provided) and
 * ends in an infinite loop.
 */
static void FreeRTOS_Error(const portCHAR* msg)
{
    if ( NULL != msg )
    {
        vDirectPrintMsg(msg);
    }

    for ( ; ; );
}



/* Startup function that creates and runs two FreeRTOS tasks */
int main(void)
{
    /* Initialize the UART 0: */
    uart_config(
        0,
        DEF_UART0_PORT,
        DEF_UART0_PIN_RX,
        DEF_UART0_PIN_TX,
        DEF_UART0_PCTL,
        DEF_UART0_BR,
        DEF_UART0_DATA_BITS,
        DEF_UART0_PARITY,
        DEF_UART0_STOP
    );

    uart_enableUart(0);

    /* Initialization of print related tasks: */
    if ( pdFAIL == printInit(APP_DEBUG_UART) )
    {
        FreeRTOS_Error("Initialization of print failed\r\n");
    }


    vDirectPrintMsg("= = = T E S T   A P P L I C A T I O N   S T A R T E D = = =\r\n\r\n");

    /* Initialize and start a watchdog timer: */
    if ( pdFAIL == watchdogInit( APP_WD_NR, APP_WD_TIMEOUT_MS) )
    {
        FreeRTOS_Error("Initialization of watchdog reloading failed\r\n");
    }

    /* Initialize LEDs and switch 1 */
    if ( pdFAIL == lightshowInit() )
    {
        FreeRTOS_Error("Initialization of LEDs and/or switch 1 failed\r\n");
    }

    /*
     * Create a task that periodically reloads a watchdog
     * and thus prevents resetting the board.
     */
    if ( pdPASS != xTaskCreate(wdTask, "wd", 128,
        (void*) &wdtaskPar, APP_PRIOR_WATCHDOG_RELOADING, NULL) )
    {
    	FreeRTOS_Error("Could not create a watchdog reloading task\r\n");
    }

    /* Create a debug print gate keeper task: */
    if ( pdPASS != xTaskCreate(printGateKeeperTask, "gk", 128,
         (void*) &printDebugParam, APP_PRIOR_PRINT_GATEKEEPER, NULL) )
    {
        FreeRTOS_Error("Could not create a print gate keeper task\r\n");
    }

    /* Create tasks that handles handles LEDs and the switch 1: */
    if ( pdPASS != xTaskCreate(lightshowTask, "lightshow", 128,
         (void*) &lsParam, APP_PRIOR_LIGHTSHOW, &(sw1Param.ledHandle)) )
    {
        FreeRTOS_Error("Could not create a light show task\r\n");
    }

    if ( pdPASS != xTaskCreate(sw1DsrTask, "switch 1", 128,
         (void*) &sw1Param, APP_PRIOR_SW1_DSR, NULL ))
    {
        FreeRTOS_Error("Could not create a switch 1 handling task\r\n");
    }

    /* Create a command processor task: */
    if ( pdPASS != xTaskCreate(vInvertText, "invert", 128,
        NULL, APP_PROIR_COMMAND_PROCESSOR, NULL) )
    {
        FreeRTOS_Error("Could not create a command processor task\r\n");
    }

    /* And finally create the timer task: */
    if ( pdPASS != xTaskCreate(vPeriodicTimerFunction, "timer", 128, (void*) &timerParam,
                               APP_PRIOR_FIX_FREQ_PERIODIC, NULL) )
    {
        FreeRTOS_Error("Could not create timer task\r\n");
    }

    vDirectPrintMsg("Enter a text via UART 1.\r\n");
    vDirectPrintMsg("It will be displayed inverted when 'Enter' is pressed.\r\n\r\n");
    vDirectPrintMsg("Press switch 1 to pause/resume light show.\r\n");

    /* Start the FreeRTOS scheduler */
    vTaskStartScheduler();

    /*
     * If all goes well, vTaskStartScheduler should never return.
     * If it does return, typically not enough heap memory is reserved.
     */

    FreeRTOS_Error("Could not start the scheduler!!!\r\n");

    /* just in case if an infinite loop is somehow omitted in FreeRTOS_Error */
    for ( ; ; );

    /* this point should never be reached but the function officially returns an int */
    return 0;
}
