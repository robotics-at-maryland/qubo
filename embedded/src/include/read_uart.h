#ifndef _READINFO_H_
#define _READINFO_H_

#include <stdbool.h>
#include <stdint.h>

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>

// The max size a message sent from computer to MCU can be
#define INPUT_BUFFER_SIZE 10

// Backspace code, hopefully
#define CODE_BS ( 0x08 )
// Enter (CR):
#define CODE_CR ( 0x0D )

// This message isn't defined in main because it's for communication between an interrupt and a task
typedef struct _read_uart_msg
{
  int32_t size;
  int32_t buffer[INPUT_BUFFER_SIZE];
} read_uart_msg;

// Queue that will send to the handler task, which will be blocking on this queue
//extern binary semaphore that will unblock the specific task

static QueueHandle_t uart_input;

// Triggered on a UART interrupt.
void _read_uart_handler(void);

void read_uart_task(void* params);

#endif
