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
 * Declaration of public functions that handle
 * the board's UART controllers.
 *
 * @author Jernej Kovacic
 */

#ifndef _UART_H_
#define _UART_H_

#include <stdint.h>


/**
 * An enumeration with supported UART baud rates.
 */
typedef enum _baud_rate_t
{
    BR_9600,
    BR_19200,
    BR_38400,
    BR_57600,
    BR_115200
} baud_rate_t;


/**
 * An enumeration with supported parities
 */
typedef enum _parity_t
{
    PAR_NONE,       /** no parity */
    PAR_ODD,        /** odd parity */
    PAR_EVEN,       /** even parity */
    PAR_STICKY_0,   /** transmits the parity bit and check as 0 */
    PAR_STICKY_1    /** transmits the parity bit and check as 1 */
} parity_t;


/**
 * An enum with possible levels when Rx interrupt
 * is triggered
 */
typedef enum _rx_interrupt_fifo_level_t
{
    RXFIFO_1_8_FULL,      /** Rx FIFO >= 1/8 full */
    RXFIFO_1_4_FULL,      /** Rx FIFO >= 1/4 full */
    RXFIFO_1_2_FULL,      /** Rx FIFO >= 1/2 full */
    RXFIFO_3_4_FULL,      /** Rx FIFO >= 3/4 full */
    RXFIFO_7_8_FULL       /** Rx FIFO >= 7/8 full */
} rx_interrupt_fifo_level_t;



void uart_enableUart(uint8_t nr);

void uart_disableUart(uint8_t nr);

void uart_flushTxFifo(uint8_t nr);

void uart_enableRx(uint8_t nr);

void uart_disableRx(uint8_t nr);

void uart_enableTx(uint8_t nr);

void uart_disableTx(uint8_t nr);

void uart_enableRxIntr(uint8_t nr);

void uart_disableRxIntr(uint8_t nr);

void uart_clearRxIntr(uint8_t nr);

void uart_characterMode(uint8_t nr);

void uart_fifoMode(uint8_t nr, rx_interrupt_fifo_level_t level);

void uart_enableNvicIntr(uint8_t nr);

void uart_disableNvicIntr(uint8_t nr);

void uart_setIntrPriority(uint8_t nr, uint8_t pri);

char uart_readChar(uint8_t nr);

void uart_printStr(uint8_t nr, const char* str);

void uart_printCh(uint8_t nr, char ch);

void uart_config(
        uint8_t nr,
        uint8_t gp,
        uint8_t pinRx,
        uint8_t pinTx,
        uint8_t pctl,
        baud_rate_t br,
        uint8_t data_bits,
        parity_t parity,
        uint8_t stop );

#endif   /* _UART_H_ */
