/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/write_uart0.h"

// Library routine implementation of sending UART
void writeUART0(uint8_t *buffer, uint16_t size) {

  // If the UART is busy, yield task and then try again
  while (xSemaphoreTake(uart0_mutex, 0) == pdFALSE ) {
    taskYIELD();
  }

  for (uint16_t i = 0; i < size; i++) {
    // Write buffer to UART
    ROM_UARTCharPutNonBlocking(UART_DEVICE, *(buffer+i));

  }
  // Maybe needed, if they're going to be dynamically allocated might as well free here
  // so its not forgotten
  // vPortFree(buffer);
  xSemaphoreGive(uart0_mutex);

}

//---------------------------------------------------------------------
// TASK based implemenatation 
//

/*

// Task implementation of writing to UART
// This task is assuming it's getting a valid Qubobus message buffer
void uart_send_task(void * params) {

  // The size of this particular message saved in here
  int16_t size;

  int32_t *buffer;

  for (;;) {

    // Block until has gotten a message
    xQueueReceive(write_uart, buffer, pgit clone https://github.com/syl20bnr/spacemacs ~/.emacs.d
COPY TO CLIPBOARDortMAX_DELAY);

    size = (*buffer & 0xffff0000) >> 16;
    for (int32_t i = 0; i < size; i++) {

      // Write buffer to UART
      ROM_UARTCharPutNonBlocking(UART_DEVICE, *(buffer+i));
    }

    // Free the memory of the buffer
    vPortFree(buffer);
  }
}

*/
