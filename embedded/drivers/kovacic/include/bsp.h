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
 * Definitions of base addresses and IRQs of Texas Instruments TM4C123GH6PMI.
 *
 * The header should be included into each source file that implements peripherals' drivers
 * or handle their interrupt requests.
 *
 * For more details, see :
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */


/*
 * At the moment, this header file is maintained manually.
 * Ideally, one day it will be generated automatically by scripts that
 * read data from BSP (board support package) databases.
 */


/*
 * Where multiple controllers of the same type are provided, their base addresses and IRQs
 * are defined as arrays. For IRQs this is pretty straightforward and arrays are simply defined
 * as values within curly brackets (e.g. "{ 3, 5, 8}"). That way it is simple to initialize
 * a statically declared array in source files where IRQs are necessary.
 * However, arrays of base addresses are a bit more tricky as addresses should be casted to
 * appropriate pointer types when assigned to pointers. This can be achieved using so called
 * "for-each macros". Within a macro definition, all addresses are enumerated as arguments to
 * another macro, e.g. CAST. Macros that are replaced by "CAST" are then defined in source files
 * when they are actually casted. For more details about this trick, see:
 * http://cakoose.com/wiki/c_preprocessor_abuse
 */


#ifndef _BSP_H_
#define _BSP_H_


/* System controller (page 231 of the Data Sheet): */
#define BSP_SYSCTL_BASE_ADDRESS           ( 0x400FE000 )

/* ----------------------------------------------- */


/*
 * Various sections Peripheral register
 * (pp. 134 -137 of the Data Sheet):
 */
#define BSP_PERIPHERAL_BASE_ADDRESS       ( 0xE000E000 )
#define BSP_SYSTICK_OFFSET                ( 0x00000010 )
#define BSP_NVIC_OFFSET                   ( 0x00000100 )
#define BSP_SCB_OFFSET                    ( 0x00000008 )
#define BSP_FPU_OFFSET                    ( 0x00000D88 )
#define BSP_SYSTICK_BASE_ADDRESS          ( BSP_PERIPHERAL_BASE_ADDRESS + BSP_SYSTICK_OFFSET )
#define BSP_NVIC_BASE_ADDRESS             ( BSP_PERIPHERAL_BASE_ADDRESS + BSP_NVIC_OFFSET )
#define BSP_SCB_BASE_ADDRESS              ( BSP_PERIPHERAL_BASE_ADDRESS + BSP_SCB_OFFSET )
#define BSP_FPU_BASE_ADDRESS              ( BSP_PERIPHERAL_BASE_ADDRESS + BSP_FPU_OFFSET )

/* ----------------------------------------------- */


/*
 * GPIO is controlled via 6 ports (port A to port F).
 * The ports may be accessed either via legacy Advanced Peripheral Bus (APB)
 * or via Advanced High-Performance Bus (AHB) addresses. Both address groups
 * are provided, the application should choose one of them and optionally
 * enable the AHB mode at the System Controller.
 */

#define BSP_NR_GPIO_PORTS   ( 6 )

/*
 * APB base addresses of all GPIO ports.
 * See pp. 658 - 659 of the Data Sheet.
 */
#define BSP_GPIO_BASE_ADDRESSES_APB(CAST) \
    CAST( 0x40004000 ) \
    CAST( 0x40005000 ) \
    CAST( 0x40006000 ) \
    CAST( 0x40007000 ) \
    CAST( 0x40024000 ) \
    CAST( 0x40025000 )

/*
 * AHB base addresses of all GPIO ports.
 * See pp. 658 - 659 of the Data Sheet.
 */
#define BSP_GPIO_BASE_ADDRESSES_AHB(CAST) \
    CAST( 0x40058000 ) \
    CAST( 0x40059000 ) \
    CAST( 0x4005A000 ) \
    CAST( 0x4005B000 ) \
    CAST( 0x4005C000 ) \
    CAST( 0x4005D000 )

/* IRQs of all GPIO ports, see pp. 104 - 105 of the Data Sheet: */
#define BSP_GPIO_IRQS   { 0, 1, 2, 3, 4, 30 }

/* ----------------------------------------------- */


/* The controller contains 8 UART interfaces: */
#define BSP_NR_UARTS        ( 8 )

/*
 * Base addresses of all UARTs.
 * See pp. 904 - 905 of the Data Sheet.
 */
#define BSP_UART_BASE_ADDRESSES(CAST) \
    CAST( 0x4000C000 ) \
    CAST( 0x4000D000 ) \
    CAST( 0x4000E000 ) \
    CAST( 0x4000F000 ) \
    CAST( 0x40010000 ) \
    CAST( 0x40011000 ) \
    CAST( 0x40012000 ) \
    CAST( 0x40013000 )

/* IRQs of all UARTs, see pp. 104 - 105 of the Data Sheet: */
#define BSP_UART_IRQS       { 5, 6, 33, 59, 60, 61, 62, 63 }

/* ----------------------------------------------- */


/* The controller contains 2 watchdog timers: */
#define BSP_NR_WATCHDOGS    ( 2 )

/*
 * Base addresses of both watchdog timers.
 * See page 777 of the Data Sheet.
 */
#define BSP_WATCHDOG_BASE_ADDRESSES(CAST) \
    CAST( 0x40000000 ) \
    CAST( 0x40001000 )

/*
 * Both watchdog timers share a single IRQ,
 * see pp. 104 - 105 of the Data Sheet:
 */
#define BSP_WATCHDOG_IRQ    ( 18 )

/* ----------------------------------------------- */

#endif   /* _BSP_H_ */
