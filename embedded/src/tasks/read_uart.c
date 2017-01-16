/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "include/read_uart.h"

void initReadUART(void) {
  read_uart = xQueueCreate(Q_SIZE, sizeof(int32_t));
}

void UARTIntHandler(void) {
  uint32_t status = ROM_UARTIntStatus(UART0_BASE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART0_BASE, status);

  // Get 32 bits from the UART
  int32_t c = ROM_UARTCharGetNonBlocking(UART0_BASE);
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
  int32_t *buffer;

  for (;;) {
    // Get the ptr to the message
    xQueueReceive(read_uart, buffer, portMAX_DELAY);


  }
}
