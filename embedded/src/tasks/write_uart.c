// QSCU
#include "include/write_uart.h"
#include "include/endpoints.h"

#include <stdbool.h>
#include <stdint.h>

// FreeRTOS
#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <portable/MemMang/heap_4.c>

// Tiva
#include <inc/hw_ints.h>
#include <inc/hw_memmap.h>
#include <driverlib/debug.h>
#include <driverlib/fpu.h>
#include <driverlib/gpio.h>
#include <driverlib/interrupt.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>

// This task is assuming it's getting a valid Qubobus message buffer
void uart_send_task(void * params) {

  // The size of this particular message saved in here
  int16_t size;

  int32_t *buffer;

  for (;;) {

    // Block until has gotten a message
    xQueueReceive(send_uart, buffer, portMAX_DELAY);

    size = (*buffer & 0xffff0000) >> 16;
    for (int32_t i = 0; i < size; i++) {

      // Write buffer to UART
      ROM_UARTCharPutNonBlocking(UART0_BASE, *(buffer+i));
    }

    // Free the memory of the buffer
    vPortFree(buffer);
  }
}
