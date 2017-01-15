#ifndef _READUART_H_
#define _READUART_H_

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

// Macro that takes the first 2 bytes of the buffer
#define GET_SIZE(a) (a >> 16)

#define Q_SIZE 100

static QueueHandle_t read_uart;

void initReadUART(void);

// Triggered on a UART interrupt.
void UARTIntHandler(void);

void read_uart_task(void* params);

#endif
