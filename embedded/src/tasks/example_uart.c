/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "tasks/include/example_uart.h"

bool example_uart_init() {
  if ( xTaskCreate(example_uart_task, (const portCHAR *)"Example UART", 200, NULL, tskIDLE_PRIORITY + 1, NULL)
       != pdTRUE) {

    return true;
  }
  return false;

}

static void example_uart_task(void *params) {

  uint8_t rx_buffer;
  uint8_t tx_buffer = 0xFF;

  for (;;) {
    if ( xQueueReceive(read_uart1_queue, &rx_buffer, 0) == pdTRUE ) {
      blink_rgb(BLUE_LED, 1);
    }
    writeUART1(&tx_buffer, 1);
  }
}
