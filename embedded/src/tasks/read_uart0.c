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

//qubobus 
#include "qubobus.h"
#include "io.h"

// For testing purposes
//#include "lib/include/write_uart0.h"

bool read_uart0_init(void) {
    if ( xTaskCreate(read_uart0_task, (const portCHAR *)"Read UART0", 128, NULL,
                     tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
        return true;
    }
    
    IO_State state = initialize(UART0_BASE, &write_uart_wrapper, &read_queue, 1);
    #ifdef DEBUG
    UARTprintf("hey the thing worked!\n");
    #endif
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

static ssize_t read_queue(void* io_host, void* buffer, size_t size){
    if(xQueueReceive(read_uart0_queue, buffer, 0) == pdTRUE){
        return 1;
    }
    else { return 0;}
    
    
}

static ssize_t write_uart_wrapper(void* io_host, void* buffer, size_t size){
    writeUART0(buffer, size);
    return size;  //sg: this return needs to change at some point for sure. 

}
