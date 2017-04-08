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


    IO_State state = initialize(UART0_BASE, &read_queue, &write_uart_wrapper, 10);

    #ifdef DEBUG
    UARTprintf("qubobus initialized!\n");
    #endif

    //int error = wait_connect(&state);


    #ifdef DEBUG
    //UARTprintf("connected\n");
    //UARTprintf("error reads as %i\n", error);
    #endif


    return false;
}



static void read_uart0_task(void* params) {

    // Qubobus driver code to assemble/interpret messages here
    uint8_t buffer;

    #ifdef DEBUG
    UARTprintf("began reading task\n");
    #endif

    write_uart_wrapper(NULL, "This is a longer print message to test the UART\r\n", 49);
    write_uart_wrapper(NULL, "an even longer message to print to make sure the UART can print over 64 characters\r\n", 84);

    for (;;) {
        if ( xQueueReceive(read_uart0_queue, &buffer, 0) == pdPASS ) {

            #ifdef DEBUG
            UARTprintf("Got %d\n", buffer);
            //blink_rgb(BLUE_LED, 1);
            #endif

        }
        vTaskDelay(25 / portTICK_RATE_MS);
        #ifdef DEBUG
        UARTprintf("reading return\n");
        #endif
        return;
    }
}

static ssize_t read_queue(void* io_host, void* buffer, size_t size){
    if(xQueueReceive(read_uart0_queue, buffer, 0) == pdPASS){
        return 1;
    }
    else { return 0;}


}
