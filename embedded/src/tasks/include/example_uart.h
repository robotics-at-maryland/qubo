/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _EXAMPLE_UART_H_
#define _EXAMPLE_UART_H_

#include "include/read_uart0_queue.h"
#include "include/read_uart1_queue.h"
#include "lib/include/write_uart.h"
#include "lib/include/rgb.h"

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

volatile QueueHandle_t read_uart0_queue;
volatile QueueHandle_t read_uart1_queue;

bool example_uart_init(void);

static void example_uart_task(void *params);

#endif
