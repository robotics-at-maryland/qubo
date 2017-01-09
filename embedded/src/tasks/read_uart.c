/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

// QSCU
#include "include/read_uart.h"
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


void _read_uart_handler(void) {
  uint32_t status = UARTIntStatus(UART0_BASE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART0_BASE, status);

  // Get the first 32 bits, to get the size of message
  int32_t first_32 = ROM_UARTCharGetNonBlocking(UART0_BASE);

  // Bitmask first half of first message to get size
  int16_t size = ((first_32 & 0xffff0000) >> 16);

  // NEED TO DYNAMICALLY ALLOCATE THIS, SO PTR DOESNT DIE
  int32_t *message = pvPortMalloc(size*sizeof(int32_t));

  message[0] = first_32;

  // Copy the whole message into array
  for (int32_t i = 0; i < size; i++) {
    if (!ROM_UARTCharsAvail(UART0_BASE)) {
      // ERROR CONDITION
    }
    int32_t ch = ROM_UARTCharGetNonBlocking(UART0_BASE);

    // Optional LED blink to show its receiving
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, GPIO_PIN_2);

    //
    // Delay for 1 millisecond.  Each SysCtlDelay is about 3 clocks.
    //
    SysCtlDelay(SysCtlClockGet() / (1000 * 3));

    //
    // Turn off the LED
    //
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0);

    // Write character to buffer
    *(message+i) = ch;
  }

  // Send buffer to task to handle message
  xQueueSendToBackFromISR(read_uart, message, NULL);
}


void read_uart_task(void* params) {


  for (;;) {
    xQueueRecieve(read_uart, )

  }
}
