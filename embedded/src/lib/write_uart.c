#include "include/write_uart.h"

// Creates the mutex
void initUARTWrite(void) {
  uart_mutex = xSemaphoreCreateMutex();
}

// Library routine implementation of sending UART
boolean UARTWrite(int32_t *buffer, int16_t size) {
  if ( xSemaphoreTake(uart_mutex, 0) ) {
    for (int16_t i = 0; i < size; i++) {
      // Write buffer to UART
      ROM_UARTCharPutNonBlocking(UART0_BASE, *(buffer+i));

    }
    // Maybe needed, if they're going to be dynamically allocated might as well free here
    // so its not forgotten
    // vPortFree(buffer);
    return true;
  }
  // Mutex is busy, not really sure if this should be handled here or the calling function
  return false;

  /**
     Could have something like this on calling:

     boolean status = false;
     while(!status) {
     if( !(status = UARTWrite()))
     // uart is busy right now, so yield and try again
     yield scheduler
     }

  */

}

//---------------------------------------------------------------------
// TASK based implemenatation 
//

// Task implementation of writing to UART
// This task is assuming it's getting a valid Qubobus message buffer
void uart_send_task(void * params) {

  // The size of this particular message saved in here
  int16_t size;

  int32_t *buffer;

  for (;;) {

    // Block until has gotten a message
    xQueueReceive(write_uart, buffer, portMAX_DELAY);

    size = (*buffer & 0xffff0000) >> 16;
    for (int32_t i = 0; i < size; i++) {

      // Write buffer to UART
      ROM_UARTCharPutNonBlocking(UART0_BASE, *(buffer+i));
    }

    // Free the memory of the buffer
    vPortFree(buffer);
  }
}
