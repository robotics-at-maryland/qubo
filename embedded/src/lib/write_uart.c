/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/write_uart.h"

// Library routine implementation of sending UART
void writeUART0(uint8_t *buffer, uint16_t size) {

  // If the UART is busy, yield task and then try again
  while (xSemaphoreTake(uart0_mutex, 0) == pdFALSE ) {
    taskYIELD();
  }

  for (uint16_t i = 0; i < size; i++) {
    // Write buffer to UART
    ROM_UARTCharPutNonBlocking(UART0_BASE, *(buffer+i));

  }
  // Maybe needed, if they're going to be dynamically allocated might as well free here
  // so its not forgotten
  // vPortFree(buffer);
  xSemaphoreGive(uart0_mutex);

}

// Library routine implementation of sending UART
void writeUART1(uint8_t *buffer, uint16_t size) {

  // If the UART is busy, yield task and then try again
  while (xSemaphoreTake(uart1_mutex, 0) == pdFALSE ) {
    taskYIELD();
  }

  for (uint16_t i = 0; i < size; i++) {
    // Write buffer to UART
    ROM_UARTCharPutNonBlocking(UART1_BASE, *(buffer+i));

  }
  // Maybe needed, if they're going to be dynamically allocated might as well free here
  // so its not forgotten
  // vPortFree(buffer);
  xSemaphoreGive(uart1_mutex);
}


