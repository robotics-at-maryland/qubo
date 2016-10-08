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
 * Implementation of functions that handle data receiving via a UART.
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
#include "uart.h"
#include "bsp.h"
#include "receive.h"



/* A constant denoting that UART number is not specified */
#define UART_UNSPECIFIED        ( (uint8_t) -1 )


/* Numeric codes for special keys: */

/* This code is received when BackSpace is pressed: */
#define CODE_BS                 ( 0x08 )
/* Enter (CR): */
#define CODE_CR                 ( 0x0D )


/*
 * Total length of a string buffer:
 * APP_RECV_BUFFER_SIZE + additional 3 characters for "\r\n\0"
 */
#define RECV_TOTAL_BUFFER_LEN        ( APP_RECV_BUFFER_LEN + 2 + 1 )


/* Queues for received characters that have not been processed yet */
static QueueHandle_t recvQueue[ BSP_NR_UARTS ] =
    { [ 0 ... (BSP_NR_UARTS-1) ] = NULL };
    /*
     * A GCC extension that initializes all array's
     * elements to NULL. See also
     * https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Designated-Inits.html
     */


/**
 * Initializes all receive related tasks and synchronization primitives.
 * This function must be called before anything is attempted to be received!
 *
 * @param uart_nr - number of the UART
 *
 * @return pdPASS if initialization is successful, pdFAIL otherwise
 */
int16_t recvInit(uint8_t uart_nr)
{
    /* Check if UART number is valid */
    if ( uart_nr >= BSP_NR_UARTS )
    {
        return pdFAIL;
    }

    /*
     * Do nothing (return pdPASS) if the uartNr
     * has already been configured.
     */
    if ( NULL != recvQueue[uart_nr] )
    {
        return pdPASS;
    }

    /* Create and assert a queue for received characters from selected UART */
    recvQueue[uart_nr] = xQueueCreate(APP_RECV_QUEUE_SIZE, sizeof(portCHAR));
    if ( NULL == recvQueue[uart_nr] )
    {
        return pdFAIL;
    }

    /* Enable the UART's IRQ on NVIC */
    uart_enableNvicIntr(uart_nr);
    uart_setIntrPriority(uart_nr, APP_DEF_UART_IRQ_PRIORITY);

    /* Wait until the UART's Tx FIFO is empty: */
    uart_flushTxFifo(uart_nr);

    /* Configure the UART to receive data and trigger interrupts on receive */
    uart_enableRx( uart_nr );
    uart_enableRxIntr( uart_nr );

    return pdPASS;
}

/*
 * An "unofficial" function (i.e. it should only be called from
 * handlers.c and is therefore not publicly exposed in receive.h)
 * handles IRQ requests by UARTs. It reads a character from the
 * selected UART and pushes it into the designated queue.
 *
 * @param uart - number of the UART
 */
void _recv_intHandler(uint8_t uart)
{
    portCHAR ch;

    /*
     * Since this function is not "public" it could rely that
     * 'uart' is valid. Anyway a check is performed, just in case...
     */
    if ( uart < BSP_NR_UARTS )
    {
        /* Read the received character from the UART's FIFO */
        ch = uart_readChar(uart);

        /*
         * Push it to the queue.
         * Note, since this is not a FreeRTOS task,
         * a *FromISR implementation of the command must be called!
         */
        xQueueSendToBackFromISR(recvQueue[uart], (void*) &ch, pdFALSE);

        /* And acknowledge the interrupt on the UART controller */
        uart_clearRxIntr(uart);
    }
}


/**
 * A FreeRTOS task that processes received characters.
 * The task is waiting in blocked state until the ISR handler pushes something
 * into the queue. If the received character is valid, it will be appended to a
 * string buffer. When 'Enter' is pressed, the entire string will be sent to UART0.
 *
 * @param params - (void*) casted pointer to an instance of recvUartParam with the selected UART number
 */
void recvTask(void* params)
{
    const recvUartParam* const par = (recvUartParam*) params;
    const uint8_t recvUartNr =
        ( NULL != par ? par->uartNr : UART_UNSPECIFIED );

    /* A queue to send a pointer to the current string buffer: */
    const QueueHandle_t queue =
        ( NULL!=par && NULL!=par->queue ? *(par->queue) : NULL);

    portCHAR ch;

    /*
     * Allocated "circular" buffer.
     *
     * Note: As this task is not likely to be deleted, it is OK
     * to allocate a buffer here (in stack). If heap space is scarce
     * or the task is more likely to be deleted before buffers are
     * read by another task, then the buffers must be declared
     * as global variables!
     */
    portCHAR buf[ APP_RECV_BUFFER_SIZE ][ RECV_TOTAL_BUFFER_LEN ];
    uint16_t i;
    uint16_t bufCntr = 0;
    uint16_t bufPos = 0;
    portCHAR* currentBuf;

    /* Init the buffer with '\0' characters */
    for ( i=0; i<APP_RECV_BUFFER_SIZE; ++i )
    {
        memset((void*) buf[i], '\0', RECV_TOTAL_BUFFER_LEN);
    }

    /*
     * If it was not possible to obtain a valid UART number,
     * the task will end and delete immediately.
     */
    if ( NULL != queue &&
         UART_UNSPECIFIED != recvUartNr &&
         recvUartNr < BSP_NR_UARTS &&
         NULL != recvQueue[recvUartNr] )
    {
        currentBuf = buf[bufCntr];

        for ( ; ; )
        {
            /* The task is blocked until something appears in the queue */
            xQueueReceive(recvQueue[recvUartNr], (void*) &ch, portMAX_DELAY);

            /*
             * In GCC, it is allowed to specify a range of consecutive
             * values in a single case label:
             * https://gcc.gnu.org/onlinedocs/gcc/Case-Ranges.html
             */
            switch (ch)
            {
                /* "Ordinary" valid characters that will be appended to a buffer */

                /* Uppercase letters 'A' .. 'Z': */
                case 'A' ... 'Z' :

                /* Lowercase letters 'a'..'z': */
                case 'a' ... 'z' :

                /* Decimal digits '0'..'9': */
                case '0' ... '9' :

                /* Other valid characters: */
                case ' ' :
                case '_' :
                case '+' :
                case '-' :
                case '/' :
                case '.' :
                case ',' :
                {
                    if ( bufPos < APP_RECV_BUFFER_LEN )
                    {
                        /* If the buffer is not full yet, append the character */
                        currentBuf[ bufPos ] = ch;
                        /* and increase the position index: */
                        ++bufPos;
                    }

                    break;
                }

                /* Backspace must be handled separately: */
                case CODE_BS :
                {
                    /*
                     * If the buffer is not empty, decrease the position index,
                     * i.e. "delete" the last character
                     */
                    if ( bufPos>0 )
                    {
                        --bufPos;
                    }

                    break;
                }

                /* 'Enter' a.k.a. Carriage Return (CR): */
                case CODE_CR :
                {
                    /* Append characters to terminate the string:*/
                    currentBuf[bufPos] = '\0';
                    /* Send the the string's pointer to the queue: */
                    xQueueSendToBack(queue, (void*) &currentBuf, 0);
                    /* And switch to the next line of the "circular" buffer */
                    ++bufCntr;
                    bufCntr %= APP_RECV_BUFFER_SIZE;
                    currentBuf = buf[bufCntr];
                    /* "Reset" the position index */
                    bufPos = 0;

                    break;
                }

            }  /* switch */

        }  /* for */
    } /* if */

    vTaskDelete(NULL);
}
