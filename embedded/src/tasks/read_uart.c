/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "tasks/include/read_uart.h"

// For testing purposes
#include "lib/include/write_uart.h"

bool read_uart_init(void) {
  //  UARTIntRegister(UART_DEVICE, UARTIntHandler);

  if ( xTaskCreate(read_uart_task, (const portCHAR *)"Read UART", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}



static void read_uart_task(void* params) {

  // Qubobus driver code to assemble/interpret messages here
  uint8_t buffer[8] = "Recieved";

  for (;;) {

    if ( xQueueReceive(read_uart_queue, buffer, 0) == pdTRUE ) {
      rgb_on(BLUE_LED);
      writeUART(buffer, 8);
      rgb_off(BLUE_LED);
    }
  }
}
