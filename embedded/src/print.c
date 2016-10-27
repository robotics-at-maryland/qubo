/*
Copyright 2013, Jernej Kovacic

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
 * Implementation of functions that perform printing messages to a UART
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>

#include "print.h"
#include "bsp.h"
#include "uart.h"



/*
 * A string buffer is necessary for printing individual characters:
 * (1) The gate keeper tasks accepts pointers to strings only. Hence a character will
 *     be placed into a short string, its first character will be the actual character,
 *     followed by '\0'.
 * (2) Corruptions must be prevented when several tasks call vPrintChar simultaneously.
 *     To accomplish this, the buffer will consist of several strings, the optimal number
 *     depends on the application.
 */

/* A constant denoting that UART number is not specified */
#define UART_UNSPECIFIED        ( (uint8_t) -1 )

/* The number of actual strings for the buffer has been defined in "FreeRTOSConfig.h" */

/* Length of one buffer string, one byte for the character, the other one for '\0' */
#define CHR_BUF_STRING_LEN      ( 2 )

/* Allocate the buffer for printing individual characters */
static portCHAR printChBuf[ APP_PRINT_CHR_BUF_SIZE ][ CHR_BUF_STRING_LEN ];

/* Position of the currently available "slot" in the buffer */
static uint16_t chBufCntr = 0;


/* Messages to be printed will be pushed to these queues */
static QueueHandle_t printQueue[ BSP_NR_UARTS ] =
    { [ 0 ... (BSP_NR_UARTS-1) ] = NULL };
    /*
     * A GCC extension that initializes all array's
     * elements to NULL. See also
     * https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Designated-Inits.html
     */



/**
 * Initializes all print related tasks and synchronization primitives.
 * This function must be called before anything is attempted to be printed
 * via vPrintMsg or vPrintChar!
 *
 * @param uart_nr - UART number to configure tasks for
 *
 * @return pdPASS if initialization is successful, pdFAIL otherwise
 */
int16_t printInit(uint8_t uart_nr)
{
    uint16_t i;

    /*
     * Initialize the character print buffer.
     * It is sufficient to set each string's second character to '\0'.
     */
    for ( i=0; i<APP_PRINT_CHR_BUF_SIZE; ++i )
    {
        printChBuf[i][1] = '\0';
    }

    chBufCntr = 0;

    /* Check if UART number is valid */
    if ( uart_nr >= BSP_NR_UARTS )
    {
        return pdFAIL;
    }

    /*
     * Do nothing (return pdPASS) if the uartNr
     * has already been configured.
     */
    if ( NULL != printQueue[uart_nr] )
    {
        return pdPASS;
    }

    /* Create and assert a queue for the gate keeper task */
    printQueue[uart_nr] = xQueueCreate(APP_PRINT_QUEUE_SIZE, sizeof(portCHAR*));
    if ( NULL == printQueue[uart_nr] )
    {
        return pdFAIL;
    }

    /* Wait until the UART's Tx FIFO is empty: */
    uart_flushTxFifo(uart_nr);

    /* Enable the UART for transmission */
    uart_enableTx(uart_nr);

    return pdPASS;
}


/**
 * A gate keeper task that waits for messages to appear in the print queue
 * for its selected UART number and prints them. This prevents corruption
 * of printed messages if a task that actually attempts to print, is preempted.
 *
 * If it is not possible to obtain a valid UART number or the UART has
 * not been initialized yet, the task will end immediately and will
 * be deleted.
 *
 * @param params - (void*) casted pointer to an instance of printUartParam with the selected UART number
 */
void printGateKeeperTask(void* params)
{
    const printUartParam* const par = (printUartParam*) params;
    const uint8_t printUartNr =
        ( NULL != par ? par->uartNr : UART_UNSPECIFIED );

    portCHAR* message;

    /*
     * If it was not possible to obtain a valid UART number,
     * the task will end and delete immediately.
     */
    if ( UART_UNSPECIFIED != printUartNr &&
         printUartNr < BSP_NR_UARTS &&
         NULL != printQueue[printUartNr] )
    {
        for ( ; ; )
        {
            /* The task is blocked until something appears in the queue */
            xQueueReceive(printQueue[printUartNr], (void*) &message, portMAX_DELAY);
            /* Print the message in the queue */
            uart_printStr(printUartNr, message);
        }
    }

    vTaskDelete(NULL);
}


/**
 * Prints a message in a thread safe manner - even if the calling task is preempted,
 * the entire message will be printed.
 *
 * Nothing is printed if 'uart' is invalid or not initialized or if'msg' equals NULL.
 *
 * @note This function may only be called when the FreeRTOS scheduler is running!
 *
 * @param uart - UART number where the message will be printed
 * @param msg - a message to be printed
 */
void vPrintMsg(uint8_t uart, const portCHAR* msg)
{
    if ( uart < BSP_NR_UARTS &&
         NULL != printQueue[uart] &&
         NULL != msg )
    {
        xQueueSendToBack(printQueue[uart], (void*) &msg, 0);
    }
}


/**
 * Prints a character in a thread safe manner - even if the calling task preempts
 * another printing task, its message will not be corrupted. Additionally, if another
 * task attempts to print a character, the buffer will not be corrupted.
 *
 * Nothing is printed if 'uart' is invalid or not initialized.
 *
 * @note This function may only be called when the FreeRTOS scheduler is running!
 *
 * @param uart - UART number where the character will be printed
 * @param ch - a character to be printed
 */
void vPrintChar(uint8_t uart, portCHAR ch)
{
    /*
     * If several tasks call this function "simultaneously", the buffer may get
     * corrupted. To prevent this, the buffer contains several strings
     */

    /* Return immediately if uart is invalid or not configured */
    if ( uart>=BSP_NR_UARTS || NULL==printQueue[uart] )
    {
        return;
    }

    /*
     * Put 'ch' to the first character of the current buffer string,
     * note that the second character has been initialized to '\0'.
     */
    printChBuf[chBufCntr][0] = ch;

    /* Now the current buffer string may be sent to the printing queue */
    vPrintMsg(uart, printChBuf[chBufCntr]);

    /*
     * Update chBufCntr and make sure it always
     * remains between 0 and CHR_PRINT_BUF_SIZE-1
     */
    ++chBufCntr;
    chBufCntr %= APP_PRINT_CHR_BUF_SIZE;
}


/**
 * Prints a message directly to the debug UART (defined in
 * FreeRTOSConfig.h, typically 0). The function is not thread
 * safe and corruption is possible when multiple tasks attempt
 * to print "simultaneously"
 *
 * Nothing is printed if 'msg' equals NULL.
 *
 * @note This function should only be called when the FreeRTOS
 *       scheduler is not running!
 *
 * @param msg - a message to be printed
 */
void vDirectPrintMsg(const portCHAR* msg)
{
    if ( NULL != msg )
    {
        uart_printStr(APP_DEBUG_UART, msg);
    }
}


/**
 * Prints a character directly to the debug UART (defined in
 * FreeRTOSConfig.h, typically 0). The function is not thread
 * safe and corruption is possible when multiple tasks attempt
 * to print "simultaneously".
 *
 * @note his function should only be called when the FreeRTOS
 *       scheduler is not running!
 *
 * @param ch - a character to be printed
 */
void vDirectPrintCh(portCHAR ch)
{
    uart_printCh(APP_DEBUG_UART, ch);
}
