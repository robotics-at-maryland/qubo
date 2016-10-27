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
 * Declaration of functions that handle printing via a UART.
 *
 * @author Jernej Kovacic
 */


#ifndef _PRINT_H_
#define _PRINT_H_

#include <stdint.h>

#include <FreeRTOS.h>


/**
 * A struct with parameters to be passed to the
 * task that prints messages to UART(s)
 */
typedef struct _printUartParam
{
    uint8_t uartNr;     /** UART number */
} printUartParam;


int16_t printInit(uint8_t uart_nr);

void printGateKeeperTask(void* params);

void vPrintMsg(uint8_t uart, const portCHAR* msg);

void vPrintChar(uint8_t uart, portCHAR ch);

void vDirectPrintMsg(const portCHAR* msg);

void vDirectPrintCh(portCHAR ch);


#endif  /* _PRINT_H_ */
