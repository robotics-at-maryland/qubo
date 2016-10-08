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
 * Implementation of functions that interact with the
 * selected watchdog timer, e.g. configuration and reloading.
 *
 * The file also includes a simple ISR that resets the board
 * if the watchdog reached timeout.
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>

#include <FreeRTOS.h>
#include <task.h>

#include "wdtask.h"
#include "watchdog.h"
#include "bsp.h"
#include "scb.h"


#define WD_UNSPECIFIED       ( (uint8_t) -1 )

/* A constant indicating that no watchdog has been selected yet */
static uint8_t __wdNr = WD_UNSPECIFIED;


/*
 * Selected watchdog's interrupt handler.
 *
 * It resets the board.
 */
static void __wdHandler(void)
{
    scb_reset();

    /*
     * The interrupt flag is intentionally not cleared.
     * If the command above does not succeed for any reason,
     * the board will automaticall reset on next timeout.
     */
}


/**
 * Configures the selected watchdog timer.
 * The watchdog is configured to reset after the first time out.
 * The function also starts the selected watchdog.
 *
 * @note Once this function successfully configures one watchdog,
 *       it cannot be called (and configure other watchdogs)
 *       anymore. In this case, nothing will be done and
 *       pdFAIL will be returned immediately.
 *
 * @param wd - desired watchdog timer (between 0 and 1)
 * @param timeoutMs -watchdog's time out period in milliseconds
 *
 * @return pdPASS on success, pdFAIL if 'wd' is invalid
 */
int16_t watchdogInit( uint8_t wd, uint32_t timeoutMs )
{
    int16_t retVal = pdFAIL;

    /*
     * Immediately calculate the required load for the selected
     * watchdog, using relevant clocks' know frequencies.
     * Note: the frequency must be divided by 1000 first, otherwise
     * the multiplication of 'timeoutMs' and a frequency may
     * exceed the uint32_t range!
     */
    const uint32_t timeoutTicks =
        ( 1==wd ?
          timeoutMs * (configPIOSC_CLOCK_HZ / 1000) :
          timeoutMs * (configCPU_CLOCK_HZ /1000) );

    /* The function only succeeds if no watchdog has been configured yet */
    if ( __wdNr==WD_UNSPECIFIED && wd<BSP_NR_WATCHDOGS )
    {
        __wdNr = wd;

        /* reset the watchdog */
        wd_reset( __wdNr );
        /* configure its timeout period */
        wd_config( __wdNr, timeoutTicks , WDEX_IRQ_RESET );
        /* register its ISR handler */
        wd_registerIntHandler( __wdNr, &__wdHandler );
        /* and start the watchdog */
        wd_start( __wdNr );
        /* finally assign it a high interrupt priority */
        wd_setIntPriority(0);

        retVal = pdPASS;
    }

    return retVal;
}


/**
 * A FreRTOS task that periodically reloads the configured watchdog.
 * If no watchdog has been configured, the task yields immediately.
 *
 * If 'pvParams' is NULL, the default period of 5 seconds will be applied.
 *
 * @param pvParameters - a pointer to an instance of wdDelayParams which includes the task's period in milliseconds
 */
void wdTask( void* pvParameters )
{
    const wdDelayParam* const param = (wdDelayParam*) pvParameters;
    const uint32_t delay = ( NULL==param ? APP_WD_TIMEOUT_MS/2 : param->delayMs );

    for ( ; ; )
    {
        /* reload the watchdog if it has been configured */
        if ( WD_UNSPECIFIED != __wdNr )
        {
            wd_reload( __wdNr );
        }

        /* block until the task's period expires */
        vTaskDelay( delay / portTICK_RATE_MS );
    }

    /* just in case the task somehow exits the infinite loop */
    vTaskDelete(NULL);
}
