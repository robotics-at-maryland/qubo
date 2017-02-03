/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _WRITEUART_H_
#define _WRITEUART_H_

#include "include/uart0_mutex.h"
#include "include/uart1_mutex.h"

#include <stdbool.h>
#include <stdint.h>

// FreeRTOS
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>
//#include <queue.h>
#include <heap_4.h>

// Tiva
#include <inc/hw_ints.h>
#include <inc/hw_memmap.h>
#include <driverlib/gpio.h>
#include <driverlib/interrupt.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>

SemaphoreHandle_t uart0_mutex;

SemaphoreHandle_t uart1_mutex;

void writeUART0(uint8_t *buffer, uint16_t size);

void writeUART1(uint8_t *buffer, uint16_t size);

#endif
