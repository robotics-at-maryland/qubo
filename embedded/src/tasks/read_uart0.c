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
#include "lib/include/rgb.h"


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
    blink_rgb(RED_LED, 1);
    #ifdef DEBUG
    UARTprintf("waiting for connect\n");
    #endif

    IO_State state = initialize(UART0_BASE, &read_queue, &write_uart_wrapper, 1);

    // Qubobus driver code to assemble/interpret messages here
    int error = wait_connect(&state);
    if(error){
        blink_rgb(RED_LED, 1);
    }else{
        blink_rgb(GREEN_LED, 1);
    }
    #ifdef DEBUG
    UARTprintf("connected, error: %i\n", error);
    #endif
    vTaskDelay(pdMS_TO_TICKS(2000));
    uint8_t buffer[QUBOBUS_MAX_PAYLOAD_LENGTH];
    Message message;
    struct Depth_Status d_s = { .depth_m = 2.71, .warning_level = 1};
    // write_uart_wrapper(NULL, "hello world", 11);
    // uncomment these to break the code above, don't know why
    read_message(&state, &message, buffer);
    message = create_response(&tDepthStatus, &d_s);
    write_message(&state, &message);
    // Message t_ann, o_ann;
    // read_message(state, &t_ann);
    // create_message(&o_ann, MT_ANNOUNCE, 0, NULL, 0);
    // write_message(state, &o_ann);

    for (;;) {


    }
}

static ssize_t read_queue(void* io_host, void* buffer, size_t size){
    uint8_t *data = buffer;
    int i = 0;
    //If the UART is busy, yield task and then try again

    while(i < size){
        if(xQueueReceive(read_uart0_queue, &data[i], 0) != pdPASS){
            blink_rgb(BLUE_LED, 1);
            return i;
        }
        i++;
    }
    blink_rgb(RED_LED, 1);
    return i;

}
