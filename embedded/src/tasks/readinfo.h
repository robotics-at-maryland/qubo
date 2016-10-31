#ifndef _READINFO_H_
#define _READINFO_H_

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>

#include <stdbool.h>
#include <stdint.h>

#include <inc/hw_memmap.h>
#include <inc/hw_ints.h>
#include <driverlib/interrupt.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>

// The max size a message sent from computer to MCU can be
#define INPUT_BUFFER_SIZE 10

// Backspace code, hopefully
#define CODE_BS ( 0x08 )
// Enter (CR):
#define CODE_CR ( 0x0D )

typedef struct _readinfo_msg
{
  int32_t size;
  int32_t[INPUT_BUFFER_SIZE] buffer;
} readinfo_msg;

// Queue that will send to the handler task, which will be blocking on this queue
//extern binary semaphore that will unblock the specific task

QueueHandle_t uart_input;

// Triggered on a UART interrupt.
void _readinfo_handler(void);

void readinfo_task(void* params);

#endif
