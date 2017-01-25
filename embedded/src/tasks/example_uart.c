#include "tasks/include/example_uart.h"

bool example_uart_init() {
  if ( xTaskCreate(example_uart_task, (const portCHAR *)"Example UART", 200, NULL, tskIDLE_PRIORITY + 1, NULL)
       != pdTRUE) {

    return true;
  }
  return false;

}

static void example_uart_task(void *params) {

  uint8_t buffer = 0xFF;

  for (;;) {
    uartWrite(&buffer, 1);
    ROM_SysCtlDelay(2000000);
  }
}
