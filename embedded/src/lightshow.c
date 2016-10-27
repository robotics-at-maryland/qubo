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
 * Implementation of functions that interact with
 * switches and LEDs.
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>

#include "FreeRTOSConfig.h"

#include "lightshow.h"
#include "led.h"
#include "switch.h"


/* Estimated switch bouncing time in milliseconds */
#define SWITCH_BOUNCE_MS           ( 50 )

/* Number of all supported triple LED states */
#define LED_STATES                 ( 7 )

/* Default period (in milliseconds) of LED switching during the light show */
#define DEFAULT_LS_PERIOD_MS       ( 1000 )

/* Combinations of LEDs for each state: */
static const uint8_t __states[ LED_STATES ] =
{
    LED_RED,
    LED_RED | LED_GREEN,
    LED_GREEN,
    LED_GREEN | LED_BLUE,
    LED_BLUE,
    LED_BLUE | LED_RED,
    LED_RED | LED_BLUE | LED_GREEN
};


/*
 * A semaphore for signalization between switch 1's
 * ISR and its deferred service routine task.
 */
static SemaphoreHandle_t sw1Smphr = NULL;

/*
 * A handle of the task that reenables switch 1's
 * interrupts after switch bouncing completes.
 */
static TaskHandle_t sw1enableHandle = NULL;


/*
 * An ISR that is triggered on falling edge of the switch 1.
 */
static void __sw1IntHandler(void)
{
    BaseType_t pxHigherPriorityTaskWoken;

    pxHigherPriorityTaskWoken = pdFALSE;
    /* Due to possible bouncing disable switch 1's interrupt first...*/
    switch_disableSwInt(1);
    /* ... and signal the deferred service routine task */
    xSemaphoreGiveFromISR(sw1Smphr, &pxHigherPriorityTaskWoken);

    /*
     * 'pxHigherPriorityTaskWoken' is not checked
     *  as there is no need for an early context switch
     *  if the DSR task is about to be activated.
     */
}


/*
 * When a switch is pressed or released, it may bounce
 * (oscillate) for a certain period of time. During this
 * period, it is recommended to disable its interrupts to
 * prevent handling of "fake" interrupt triggering events.
 *
 * When switch 1's interrupt is triggered and its interrupt
 * triggering is disabled, this task is resumed. After the
 * bouncing period completes, it reenables interrupt triggering
 * of the switch and then suspends itself until the switch 1 is
 * pressed again...
 *
 * Note that the task must be disabled immediately after its
 * creation, before the scheduler is started.
 *
 * @param params - ignored
 */
static void __enableSw1IntrTask(void* params)
{
    for ( ; ; )
    {
        /*
         * The deferred service routine task will
         * resume this task and this point will be reached.
         *
         * First delay for the bouncing period,
         */
        vTaskDelay( SWITCH_BOUNCE_MS / portTICK_RATE_MS );

        /* reenable switch 1 interrupt triggering, */
        switch_enableSwInt(1);

        /* and finally suspend the task */
        vTaskSuspend(NULL);
    }

    /* just to suppress a warning due to unused parameters */
    (void) params;
}


/**
 * Initializes everything required for a LED light show
 * and its control using the switch 1.
 *
 * After this function completes successfully, 'lightshowTask'
 * and 'sw1Task' can safely be started.
 *
 * @return pdPASS on success, pdFAIL otherwise
 */
int16_t lightshowInit(void)
{
	/*
	 * Create a semaphore for signalization between
	 * switch 1 ISR and DSR. Also check success
	 * of the creation.
	 */
    sw1Smphr = xSemaphoreCreateBinary();
    if ( NULL == sw1Smphr )
    {
        return pdFAIL;
    }

    /* Configure LED' and switches' GPIO pins: */
    led_config();
    switch_config();

    /* Disable interrupt triggering by switch 1 */
    switch_disableSwInt(1);

    /* Create a task that reenables SW1 interrupts */
    if ( pdPASS != xTaskCreate(__enableSw1IntrTask, "sw1intenable", 128,
         NULL, APP_PRIOR_SW1_REENABLE_INTR, &sw1enableHandle) )
    {
        return pdFAIL;
    }

    /* and suspend this task */
    vTaskSuspend(sw1enableHandle);

    /* Register interrupt handler for switch 1 */
    switch_registerIntrHandler(1, &__sw1IntHandler);

    return pdPASS;
}


/**
 * A task that periodically turns on and off built in LEDs.
 *
 * @param params - (void*) casted pointer to an instance of LightShowParam_t
 *                 with the period of LED switching (1000 ms if params is NULL)
 */
void lightshowTask(void* params)
{
    uint8_t currState = 0;
    uint32_t delay = DEFAULT_LS_PERIOD_MS;

    /* obtain 'delay' if provided */
    if ( NULL == params )
    {
        delay = ((LightShowParam_t*) params)->delayMs;
    }

    for ( ; ; )
    {
        /*
         * Turn off all LEDs and turn on the
         * combination, depending on state.
         */
        led_allOff();
        led_on(__states[currState]);;

        /* increment the state variable */
        ++currState;
        currState %= LED_STATES;

        vTaskDelay( delay / portTICK_RATE_MS );
    }
}


/**
 * A deferred service routine task that receives a
 * signal from switch 1 ISR and pauses/resumes the light show.
 *
 * The task is deleted immediately if 'params' equals NULL.
 *
 * @param params - (void*) casted pointer to an instance of Switch1TaskParam_t with the light show task handle
 */
void sw1DsrTask(void* params)
{
    TaskHandle_t lsTask = NULL;
	bool lsRunning = true;

    if ( NULL != params )
    {
        lsTask = ((Switch1TaskParam_t*) params)->ledHandle;
    }

    if ( NULL != lsTask )
    {
        /* Initially enable switch 1 interrupt triggering */
        switch_enableSwInt(1);

        for ( ; ; )
        {
            /* wait until a signal is received from the ISR */
            xSemaphoreTake( sw1Smphr, portMAX_DELAY);

            /* update status of the light show task */
            lsRunning = ( true==lsRunning ? false : true );

            /*
             * Depending on the updated status, either
             * suspend or resume the light show task
             */
            if ( false == lsRunning )
            {
                vTaskSuspend(lsTask);
            }
            else
            {
                vTaskResume(lsTask);
            }

            /*
             * And resume the task that will reenable
             * switch 1 interrupt triggering after the
             * bouncing period.
             */
            vTaskResume(sw1enableHandle);
        }  /* for */
    }  /* if */

    vTaskDelete(NULL);
}
