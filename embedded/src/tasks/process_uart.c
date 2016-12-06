#include "process_uart.h"

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <semaphore.h>

#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>

void process_uart_task(void *params) {

  process_uart_msg *msg = (process_uart_msg *) params;
  int32_t buffer[UART_BUFFER_SIZE];

  for (;;) {

    xQueueReceive(*(msg->read_uart_msg), &buffer, portMAX_DELAY);

  }
}
