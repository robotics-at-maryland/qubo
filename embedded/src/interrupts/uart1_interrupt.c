/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Queue for the UART interrupt
#include "interrupts/include/uart1_interrupt.h"

// Interrupt for UART
void UART1IntHandler(void) {
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
  xQueueSendToBackFromISR(read_uart1_queue, &c, NULL);
}
