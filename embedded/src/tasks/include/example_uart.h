/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _EXAMPLE_UART_H_
#define _EXAMPLE_UART_H_

#include "lib/include/write_uart.h"

// FreeRTOS
#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>
#include <semphr.h>

// Tiva
#include <stdbool.h>
#include <stdint.h>
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>

bool example_uart_init(void);

static void example_uart_task(void *params);

#endif
