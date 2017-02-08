/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "tasks/include/read_uart1.h"

// For testing purposes
//#include "lib/include/write_uart1.h"

bool read_uart1_init(void) {
  if ( xTaskCreate(read_uart1_task, (const portCHAR *)"Read UART1", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}



static void read_uart1_task(void* params) {

  // Qubobus driver code to assemble/interpret messages here
  uint8_t buffer;

  for (;;) {
    if ( xQueueReceive(read_uart1_queue, &buffer, 0) == pdTRUE ) {
      writeUART1(&buffer, 1);

      #ifdef DEBUG
      UARTprintf("Recieved on UART1\n");
      #endif

      blink_rgb(GREEN_LED, 1);
    }
    vTaskDelay(100 / portTICK_RATE_MS);
  }
}
