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
//#include "lib/include/write_uart.h"

bool read_uart0_init(void) {
    if ( xTaskCreate(read_uart0_task, (const portCHAR *)"Read UART0", 128, NULL,
                     tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
        return true;
    }



    #ifdef DEBUG
    UARTprintf("qubobus initialized!\n");
    #endif

    #ifdef DEBUG
    //UARTprintf("connected\n");
    //UARTprintf("error reads as %i\n", error);
    #endif


    return false;
}



static void read_uart0_task(void* params) {
    #ifdef DEBUG
    UARTprintf("waiting for connect\n");
    #endif

    IO_State state = initialize(UART0_BASE, &read_queue, &write_uart_wrapper, 10);

    //int error = wait_connect(&state);

    // Qubobus driver code to assemble/interpret messages here
    uint8_t buffer;

    #ifdef DEBUG
    UARTprintf("began reading task\n");
    #endif

    // int error = wait_connect(&state);
    uint8_t data = 0;
    for (;;) {
        while(read_queue(NULL, &data, 1)){
            write_uart_wrapper(NULL, &data, 1);
        }
        //vTaskDelay(25 / portTICK_RATE_MS);

    }
}

static ssize_t read_queue(void* io_host, void* buffer, size_t size){
    uint8_t *data = buffer;
    int i = 0;

    //sgillen@20175408-12:54 should maybe change this to so we don't rely on the order of the execution
    //in the while loop
    while((i < size) && (xQueueReceive(read_uart0_queue, data, 10) == pdPASS) ){
        //   vTaskDelay(25);
        i++;
    }

    
    return i;

}
