/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _READUART0_H_
#define _READUART0_H_

#include "lib/include/rgb.h"
#include "lib/include/write_uart.h"
#include "include/read_uart0_queue.h"

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

#ifdef DEBUG
#include <utils/uartstdio.h>
#endif

volatile QueueHandle_t read_uart0_queue;

bool read_uart0_init(void);

static void read_uart0_task(void* params);

#endif
