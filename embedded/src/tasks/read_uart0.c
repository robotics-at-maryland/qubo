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
#include "lib/include/uart_queue.h"


//qubobus
#include "qubobus.h"
#include "io.h"

extern struct UART_Queue uart0_queue;
extern struct UART_Queue uart1_queue;
static char buffer[QUBOBUS_MAX_PAYLOAD_LENGTH];

bool read_uart0_init(void) {
    if ( xTaskCreate(read_uart0_task, (const portCHAR *)"Read UART0", 1024, NULL,
                     tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
        return true;
    }

    return false;
}


ssize_t test_read_uart_queue(void *uart_queue, void* buffer, size_t size) {
    blink_rgb(BLUE_LED, 1);
    ssize_t ret = read_uart_queue(uart_queue, buffer, size);
    if (ret == size) {
        blink_rgb(BLUE_LED|GREEN_LED, 1);
    } else {
        blink_rgb(BLUE_LED|RED_LED, 1);
    }
    return ret;
}
ssize_t test_write_uart_queue(void *uart_queue, void* buffer, size_t size) {
    blink_rgb(GREEN_LED, 1);
    ssize_t ret = write_uart_queue(uart_queue, buffer, size);
    if (ret == size) {
        blink_rgb(BLUE_LED|GREEN_LED, 1);
    } else {
        blink_rgb(GREEN_LED|RED_LED, 1);
    }
    return ret;
}

static void read_uart0_task(void* params) {
    blink_rgb(RED_LED, 1);
    int error = 1;
    //test_write_uart_queue(&uart0_queue, "hello\n", 6);
    //test_write_uart_queue(&uart0_queue, "hello\n", 6);
    //IO_State state = initialize(&uart0_queue, test_read_uart_queue, test_write_uart_queue, 1);
    IO_State state = initialize(&uart0_queue, read_uart_queue, write_uart_queue, 1);
    #ifdef DEBUG
    test_write_uart_queue(&uart0_queue, "hello world sixteen\n", 20);
    #endif
    // Qubobus driver code to assemble/interpret messages here
fail:
    while ( wait_connect(&state, buffer) ){
        blink_rgb(RED_LED | BLUE_LED, 1);
    }

    struct Depth_Status d_s = { .depth_m = 2.71, .warning_level = 1};


    Message message;
    if (read_message(&state, &message, buffer)) {
        blink_rgb(RED_LED, 1);
        goto fail;
    }
    message = create_response(&tDepthStatus, &d_s);
    if (write_message(&state, &message)) {
        blink_rgb(RED_LED, 1);
        goto fail;
    }


    // Message t_ann, o_ann;
    // read_message(state, &t_ann);
    // create_message(&o_ann, MT_ANNOUNCE, 0, NULL, 0);
    // write_message(state, &o_ann);

    error = 0;

    for (;;) {
        if (error) {
            blink_rgb(RED_LED, 1);
        } else  {
            blink_rgb(GREEN_LED, 1);
        }
    }
}
