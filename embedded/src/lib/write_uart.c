/* Ross Baehr
 * R@M 2017
 * ross.baehr@gmail.com
 * edits by Jeremy
 */

#include "lib/include/write_uart.h"

//warning: If size is bigger than the buffer, we will overflow
ssize_t write_uart_wrapper(void* io_host, void* buffer, size_t size){

    //If the UART is busy, yield task and then try again
    while (xSemaphoreTake(uart0_read_mutex, 0) == pdFALSE ) {
      taskYIELD();
    }
    #ifdef DEBUG
    //UARTprintf("adding to queue\n");
    #endif
    //type "safety"
    const uint8_t *data = buffer;
    //counter to make sure we write everthing we're given
    int i = 0;

    //enable the interrupt to begin sending things to the UART FIFO
    ROM_UARTIntEnable(UART0_BASE, UART_INT_TX);

    //stuff data into the queue until we run out of queue or data
    while( (i < size) ){
        #ifdef DEBUG
        //UARTprintf("%c", data[i]);
        #endif
        if( (xQueueSend(write_uart0_queue, &(data[i]), 0) == pdPASS) ){
            i++;
        }else{
            taskYIELD();
        }

    }

    //We need to call this at least once to begin the write_message
    writeUART0();

    //wait for the data to be removed from the queue
    while( (uxQueueMessagesWaiting(write_uart0_queue) != 0) ){
        #ifdef DEBUG
        //UARTprintf("still printing: %d\n", _finished_writing);
        #endif
        taskYIELD();
    }
    xSemaphoreGive (uart0_read_mutex);

    return size;  //sg: this return needs to change at some point for sure.

}

// Library routine implementation of sending UART
void writeUART0() {

  uint8_t buffer;

  //disable the interrupt, there might be a race condition here
  ROM_IntDisable(INT_UART0);

  while( ROM_UARTSpaceAvail(UART0_BASE) && (xQueueReceiveFromISR(write_uart0_queue, &buffer, NULL) == pdPASS) ) {
    // Write buffer to UART
    //sg: this probably needs to be changed, we don't check the return value here.
    #ifdef DEBUG
    //UARTprintf("'%c'", buffer);
    #endif
    ROM_UARTCharPutNonBlocking(UART0_BASE, buffer);
  }

  if( xQueueIsQueueEmptyFromISR(write_uart0_queue) == pdTRUE ){
    //nothing to write, disable the interupt
    ROM_UARTIntDisable(UART0_BASE, UART_INT_TX);
    #ifdef DEBUG
    //UARTprintf("finished writing: %d\n", _finished_writing);
    #endif
  }else{
    ROM_UARTIntEnable(UART0_BASE, UART_INT_TX);
    #ifdef DEBUG
    //UARTprintf("not finished writing: %d\n", _finished_writing);
    #endif

  }
  //reenable interrupt
  ROM_IntEnable(INT_UART0);
  // Maybe needed, if they're going to be dynamically allocated might as well free here
  // so its not forgotten
  // vPortFree(buffer);

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
