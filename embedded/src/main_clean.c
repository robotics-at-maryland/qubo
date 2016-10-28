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
#include <unistd.h>

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
#include "tasks/i2ctask.h"

#include <stdarg.h>
#include <stdbool.h>

#include <inc/hw_i2c.h>
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <inc/hw_gpio.h>

#include <driverlib/rom.h>
#include <driverlib/i2c.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/uart.h>
#include <driverlib/sysctl.h>

#include <utils/uartstdio.h>
#include <utils/ustdlib.h>

#define DEF_TIMER_DELAY_SEC            ( 5 )
#define DEF_NR_TIMER_STR_BUFFERS       ( 3 )
#define DEF_RECV_QUEUE_SIZE            ( 3 )

/* Declaration of a function, implemented in nostdlib.c */
extern char* itoa(int32_t value, char* str, uint8_t radix);

static xQueueHandle globalQueue = 0;

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
	//ConfigureUART();
	ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
	ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);
  //	ROM_GPIOPinConfigure(GPIO_PA0_U0RX);
  //	ROM_GPIOPinConfigure(GPIO_PA1_U0TX);
  //	ROM_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);
	UARTStdioConfig(0, 115200, SysCtlClockGet());

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

    if ( init_i2ctask() != 0) {
      FreeRTOS_ERROR("Error in init_i2ctask\n");
    }

    if ( pdPASS != xTaskCreate(i2ctask, "i2c loopback", 128,
        NULL, APP_PROIR_COMMAND_PROCESSOR, NULL) )
    {
        FreeRTOS_Error("Could not create a command processor task\r\n");
    }


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
