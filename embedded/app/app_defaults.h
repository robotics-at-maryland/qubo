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
 * Sensible defaults for supported peripherals of
 * Texas Instruments TM4C123GLX Launchpad.
 *
 * @author Jernej Kovacic
 */


#ifndef _APP_DEFAULTS_H_
#define _APP_DEFAULTS_H_

#include "uart.h"


/*
 * Sensible defaults for all UART communications:
 * - 8 data bits
 * - no parity
 * - 1 stop bit
 */
#define DEF_UART_DATA_BITS      ( 8 )
#define DEF_UART_PARITY         ( PAR_NONE )
#define DEF_UART_STOP           ( 1 )


/*
 * Default settings for the UART0:
 * pins 0 and 1 of the GPIO port A, baud rate=115200
 */
#define DEF_UART0_PORT          ( GPIO_PORTA )
#define DEF_UART0_PIN_RX        ( 0 )
#define DEF_UART0_PIN_TX        ( 1 )
#define DEF_UART0_PCTL          ( 1 )
#define DEF_UART0_BR            ( BR_115200 )
#define DEF_UART0_DATA_BITS     DEF_UART_DATA_BITS
#define DEF_UART0_PARITY        DEF_UART_PARITY
#define DEF_UART0_STOP          DEF_UART_STOP

/*
 * Default settings for the UART1:
 * pins 0 and 1 of the GPIO port B, baud rate=115200
 */
#define DEF_UART1_PORT          ( GPIO_PORTB )
#define DEF_UART1_PIN_RX        ( 0 )
#define DEF_UART1_PIN_TX        ( 1 )
#define DEF_UART1_PCTL          ( 1 )
#define DEF_UART1_BR            ( BR_115200 )
#define DEF_UART1_DATA_BITS     DEF_UART_DATA_BITS
#define DEF_UART1_PARITY        DEF_UART_PARITY
#define DEF_UART1_STOP          DEF_UART_STOP


#endif  /* _APP_DEFAULTS_H_ */
