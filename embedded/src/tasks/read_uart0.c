/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/**
   Interrupt generates dynamically allocated buffer of messages.
   Their pointers are added to the read_uart task queue, who
   decides what to do with them
 */

#include "tasks/include/read_uart0.h"

// For testing purposes
//#include "lib/include/write_uart0.h"

bool read_uart0_init(void) {
  if ( xTaskCreate(read_uart0_task, (const portCHAR *)"Read UART0", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}



static void read_uart0_task(void* params) {

  // Qubobus driver code to assemble/interpret messages here
  uint8_t buffer;

  for (;;) {
    if ( xQueueReceive(read_uart0_queue, &buffer, 0) == pdTRUE ) {

      #ifdef DEBUG
      UARTprintf("Got %d\n", buffer);
      #endif

      blink_rgb(BLUE_LED, 1);
    }
    vTaskDelay(25 / portTICK_RATE_MS);
  }
}
