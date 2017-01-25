/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Queue for the UART interrupt
#include "interrupts/include/uart_interrupt.h"

// Interrupt for UART
void UARTIntHandler(void) {
  uint32_t status = ROM_UARTIntStatus(UART0_BASE, true);

  // Clear interrupt
  ROM_UARTIntClear(UART0_BASE, status);

  if ( !ROM_UARTCharsAvail(UART0_BASE) ) {
    // ERROR, Do something here
  }

  // Get one byte
  // Tivaware casts the byte to a int32_t for some reason, cast back to save space
  uint8_t c = (uint8_t)(ROM_UARTCharGetNonBlocking(UART0_BASE));
  // Push to the queue
  xQueueSendToBackFromISR(read_uart_queue, &c, NULL);

}
