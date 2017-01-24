/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "include/read_uart.h"

// For testing purposes
#include "include/write_uart.h"

void initReadUART(void) {
  read_uart = xQueueCreate(Q_SIZE, sizeof(uint8_t));
}

void UARTIntHandler(void) {
  uint32_t status = ROM_UARTIntStatus(UART_DEVICE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART_DEVICE, status);

  if ( !ROM_UARTCharsAvail(UART_DEVICE) ) {
    // ERROR, Do something here
  }

  // Get one byte
  // Tivaware casts the byte to a int32_t for some reason, cast back to save space
  uint8_t c = (uint8_t)(ROM_UARTCharGetNonBlocking(UART_DEVICE));
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

    bool status = false;
    while ( !status ) {
      status = uartWrite(0xFF, 1);
    }
  }
}
