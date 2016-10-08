/*
    FreeRTOS V9.0.0 - Copyright (C) 2016 Real Time Engineers Ltd.
    All rights reserved

    VISIT http://www.FreeRTOS.org TO ENSURE YOU ARE USING THE LATEST VERSION.

    This file is part of the FreeRTOS distribution.

    FreeRTOS is free software; you can redistribute it and/or modify it under
    the terms of the GNU General Public License (version 2) as published by the
    Free Software Foundation >>>> AND MODIFIED BY <<<< the FreeRTOS exception.

    ***************************************************************************
    >>!   NOTE: The modification to the GPL is included to allow you to     !<<
    >>!   distribute a combined work that includes FreeRTOS without being   !<<
    >>!   obliged to provide the source code for proprietary components     !<<
    >>!   outside of the FreeRTOS kernel.                                   !<<
    ***************************************************************************

    FreeRTOS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  Full license text is available on the following
    link: http://www.freertos.org/a00114.html

    ***************************************************************************
     *                                                                       *
     *    FreeRTOS provides completely free yet professionally developed,    *
     *    robust, strictly quality controlled, supported, and cross          *
     *    platform software that is more than just the market leader, it     *
     *    is the industry's de facto standard.                               *
     *                                                                       *
     *    Help yourself get started quickly while simultaneously helping     *
     *    to support the FreeRTOS project by purchasing a FreeRTOS           *
     *    tutorial book, reference manual, or both:                          *
     *    http://www.FreeRTOS.org/Documentation                              *
     *                                                                       *
    ***************************************************************************

    http://www.FreeRTOS.org/FAQHelp.html - Having a problem?  Start by reading
    the FAQ page "My application does not run, what could be wrong?".  Have you
    defined configASSERT()?

    http://www.FreeRTOS.org/support - In return for receiving this top quality
    embedded software for free we request you assist our global community by
    participating in the support forum.

    http://www.FreeRTOS.org/training - Investing in training allows your team to
    be as productive as possible as early as possible.  Now you can receive
    FreeRTOS training directly from Richard Barry, CEO of Real Time Engineers
    Ltd, and the world's leading authority on the world's leading RTOS.

    http://www.FreeRTOS.org/plus - A selection of FreeRTOS ecosystem products,
    including FreeRTOS+Trace - an indispensable productivity tool, a DOS
    compatible FAT file system, and our tiny thread aware UDP/IP stack.

    http://www.FreeRTOS.org/labs - Where new FreeRTOS products go to incubate.
    Come and try FreeRTOS+TCP, our new open source TCP/IP stack for FreeRTOS.

    http://www.OpenRTOS.com - Real Time Engineers ltd. license FreeRTOS to High
    Integrity Systems ltd. to sell under the OpenRTOS brand.  Low cost OpenRTOS
    licenses offer ticketed support, indemnification and commercial middleware.

    http://www.SafeRTOS.com - High Integrity Systems also provide a safety
    engineered and independently SIL3 certified version for use in safety and
    mission critical applications that require provable dependability.

    1 tab == 4 spaces!
*/

/**
 * @file
 *
 * Initially this file contained variables to
 * configure FreeRTOS according to application needs.
 *
 * The file was expanded with various other application
 * settings, not directly related to FreeRTOS.
 */


#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H


#include "pll_freq_divisors.h"


/*-----------------------------------------------------------
 * Application specific definitions.
 *
 * These definitions should be adjusted for your particular hardware and
 * application requirements.
 *
 * THESE PARAMETERS ARE DESCRIBED WITHIN THE 'CONFIGURATION' SECTION OF THE
 * FreeRTOS API DOCUMENTATION AVAILABLE ON THE FreeRTOS.org WEB SITE.
 *
 * See http://www.freertos.org/a00110.html.
 *----------------------------------------------------------*/

#define configUSE_PREEMPTION              1
#define configUSE_IDLE_HOOK               0
#define configUSE_TICK_HOOK               0
/* Timers' clock frequency is 50 MHz: */
#define configCPU_CLOCK_HZ                ( ( UBaseType_t ) 50000000 )
#define configTICK_RATE_HZ                ( ( TickType_t ) 1000 )
/* PIOSC frequency is 16 MHz: */
#define configPIOSC_CLOCK_HZ              ( ( UBaseType_t ) 16000000 )
#define configMAX_PRIORITIES              ( 5 )
#define configMINIMAL_STACK_SIZE          ( ( StackType_t ) 128 )
#define configTOTAL_HEAP_SIZE             ( ( size_t ) ( 10240 ) )
#define configMAX_TASK_NAME_LEN           ( 16 )
#define configUSE_TRACE_FACILITY          0
#define configUSE_16_BIT_TICKS            0
#define configIDLE_SHOULD_YIELD           1
#define configUSE_APPLICATION_TASK_TAG    1
#define configCHECK_FOR_STACK_OVERFLOW    0

#define configUSE_MUTEXES                 0

/* Co-routine definitions. */
#define configUSE_CO_ROUTINES             0
#define configMAX_CO_ROUTINE_PRIORITIES   ( 2 )

/*
 * Set the following definitions to 1 to include the
 * API function, or zero to exclude the API function.
 */

#define INCLUDE_vTaskPrioritySet              1
#define INCLUDE_uxTaskPriorityGet             1
#define INCLUDE_vTaskDelete                   1
#define INCLUDE_vTaskCleanUpResources         0
#define INCLUDE_vTaskSuspend                  1
#define INCLUDE_vTaskDelayUntil               1
#define INCLUDE_vTaskDelay                    1

/*
 * This is the raw value as per the Cortex-M4 NVIC.
 * Values can be 7 (lowest) to 0 (1?) (highest).
 */
#define configKERNEL_INTERRUPT_PRIORITY       7

/*
 * !!!! configMAX_SYSCALL_INTERRUPT_PRIORITY
 * must not be set to zero !!!!
 * See http://www.FreeRTOS.org/RTOS-Cortex-M3-M4.html.
 *
 * Note: Set this macro to a value between 1 and 7.
 * Functions, that use it, will shift it by 5 bits
 * as defined by the PRIMASK register!
 */
#define configMAX_SYSCALL_INTERRUPT_PRIORITY  ( 3 )

/*
 * This is the value being used as per the ST library which
 * permits 16 priority values, 0 to 15.  This must correspond
 * to the configKERNEL_INTERRUPT_PRIORITY setting.  Here 15
 * corresponds to the lowest NVIC value of 255.
 */
#define configLIBRARY_KERNEL_INTERRUPT_PRIORITY     15


/* ---------------------------------------------------------- */

/*
 * Other application settings, not directly
 * related to FreeRTOS
 */

/* By default set system clock frequency to 50 MHz: */
#define APP_SYS_CLOCK_DIV             ( DIV_FREQ_50_MHZ )

/* Main stack size in 32-bit words: */
#define APP_MAIN_STACK_SIZE_WORDS     ( 128 )

/*
 * Initial process stack size in 32-bit words.
 * Note: the stack of this size will only be used
 * during preparation tasks until FreeRTOS scheduler
 * is started. Then each task will be allocated
 * its own process stack from the heap.
 * For that reason, in most cases it will be sufficient
 * if the initial process stack is small.
 */
#define APP_PROCESS_STACK_SIZE_WORDS  ( 20 )

/*
 * Enable the Floating Point Unit and stacking of its
 * register on interrupt events.
 */
#define APP_FPU_ENABLE                0

/*
 * Enables lazy stacking of FPU registers.
 * If set to 0, full FPU stacking will be performed.
 */
#define APP_FPU_LAZY_STACKING         1

/*
 * Will GPIO registers be accessed via the Advanced High-Performance
 * Bus (AHB)?
 */
#define APP_GPIO_AHB                  1

/* The selected watchdog and its settings (timeout = 10 sec.) */
#define APP_WD_NR               ( 0 )
#define APP_WD_TIMEOUT_MS       ( 10000 )

/*
 * Debug UART number, used to display diagnostic
 * messages during development
 */
#define APP_DEBUG_UART          ( 0 )

/* Default priority of IRQ requests, triggered by UARTs */
#define APP_DEF_UART_IRQ_PRIORITY   ( 3 )

/*
 * Priorities of certain tasks.
 * Note: priorities should not be greater than configMAX_PRIORITIES - 1,
 * defined in FreeRTOSConfig.h (its default value equals 5).
 * If any priority is greater than this value, xTasCreate will
 * silently reduce it.
 */
#define APP_PRIOR_FIX_FREQ_PERIODIC       ( 3 )
#define APP_PRIOR_PRINT_GATEKEEPER        ( 1 )
#define APP_PRIOR_RECEIVER                ( 1 )
#define APP_PRIOR_WATCHDOG_RELOADING      ( 2 )
#define APP_PROIR_COMMAND_PROCESSOR       ( 2 )
#define APP_PRIOR_SW1_REENABLE_INTR       ( 4 )
#define APP_PRIOR_LIGHTSHOW               ( 3 )
#define APP_PRIOR_SW1_DSR                 ( 1 )

/* Size of the queue with pointers to strings that will be printed */
#define APP_PRINT_QUEUE_SIZE              ( 10 )

/* Number of string buffers to print individual characters */
#define APP_PRINT_CHR_BUF_SIZE            ( 5 )

/*
 * UARTs used by the application for receiving texts
 * and printing messages
 */
#define APP_PRINT_UART_NR                 ( 1 )
#define APP_RECV_UART_NR                  ( 1 )

/* Size of the queue holding received characters, that have not been processed yet. */
#define APP_RECV_QUEUE_SIZE               ( 10 )

/* Number of string buffers necessary to print received strings */
#define APP_RECV_BUFFER_SIZE              ( 3 )

/*
 * Number of characters in a buffer.
 * Note: this limit does not include '\0' and additional extra characters, necessary
 * to print the string properly.
 */
#define APP_RECV_BUFFER_LEN               ( 50 )

/*
 * Default delay (in seconds) for situations when this
 * parameter has not been provided to the  upload timer task.
 */
#define APP_TIMER_DELAY_SEC               ( 10 )

/*
 * Period (in milliseconds) of LED switching by
 * the light show task.
 */
#define APP_LIGHTSHOW_PERIOD_MS           ( 1000 )

#endif /* FREERTOS_CONFIG_H */
