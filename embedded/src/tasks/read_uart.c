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
  uint32_t status = UARTIntStatus(UART0_BASE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART0_BASE, status);

  // Get the first 32 bits, to get the size of message
  int32_t first_32 = ROM_UARTCharGetNonBlocking(UART0_BASE);

  // Bitmask first half of first message to get size
  int16_t size = ((first_32 & 0xffff0000) >> 16);

  // NEED TO DYNAMICALLY ALLOCATE THIS, SO PTR DOESNT DIE
  int32_t *message = pvPortMalloc(size*sizeof(int32_t));

  *message = first_32;

  // Copy the whole message into array
  for (int32_t i = 0; i < size; i++) {
    if (!ROM_UARTCharsAvail(UART0_BASE)) {
      // ERROR CONDITION
    }
    int32_t ch = ROM_UARTCharGetNonBlocking(UART0_BASE);

    // Optional LED blink to show its receiving
    ROM_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, GPIO_PIN_2);

    //
    // Delay for 1 millisecond.  Each SysCtlDelay is about 3 clocks.
    //
    ROM_SysCtlDelay(SysCtlClockGet() / (1000 * 3));

    //
    // Turn off the LED
    //
    ROM_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0);

    // Write character to buffer
    *(message+i) = ch;
  }

  // Send buffer to task to handle message
  xQueueSendToBackFromISR(read_uart, message, NULL);
}


void read_uart_task(void* params) {

  for (;;) {
    //xQueueRecieve(read_uart, )

  }
}
