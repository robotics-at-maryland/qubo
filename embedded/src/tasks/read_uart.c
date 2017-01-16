/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "include/read_uart.h"

void initReadUART(void) {
  read_uart = xQueueCreate(Q_SIZE, sizeof(uint8_t));
}

void UARTIntHandler(void) {
  uint32_t status = ROM_UARTIntStatus(UART0_BASE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART0_BASE, status);

  // Get one byte
  // Tivaware casts the byte to a int32_t for some reason, cast back to save space
  uint8_t c = (uint8_t)(ROM_UARTCharGetNonBlocking(UART0_BASE));
  // Push to the queue
  xQueueSendToBackFromISR(read_uart, c, NULL);

  // Optional LED blink to show its receiving
  ROM_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, GPIO_PIN_2);

  //
  // Delay for 1 millisecond.  Each SysCtlDelay is about 3 clocks.
  //
  ROM_SysCtlDelay(ROM_SysCtlClockGet() / (1000 * 3));

  //
  // Turn off the LED
  //
  ROM_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0);

  }


void read_uart_task(void* params) {

  // Qubobus driver code to assemble/interpret messages here
  uint8_t *buffer;

  for (;;) {
    // Get the ptr to the message
    xQueueReceive(read_uart, buffer, portMAX_DELAY);


  }
}
