#ifndef _WRITEUART_H_
#define _WRITEUART_H_

#include "include/uart_mutex.h"

#include <stdbool.h>
#include <stdint.h>

// FreeRTOS
#include <FreeRTOS.h>
//#include <task.h>
#include <semphr.h>
//#include <queue.h>
#include <heap_4.h>

// Tiva
#include <inc/hw_ints.h>
#include <inc/hw_memmap.h>
//#include <driverlib/debug.h>
//#include <driverlib/fpu.h>
#include <driverlib/gpio.h>
#include <driverlib/interrupt.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>

#define UART_DEVICE UART0_BASE

SemaphoreHandle_t uart_mutex;

bool writeUART(uint8_t *buffer, uint16_t size);

// ---------
// Task based implementation

//extern QueueHandle_t write_uart;

//void uart_send_task(void * params);

#endif
