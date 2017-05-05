/*
 * Greg Harris
 * R@M 2017
 * gharris1727@gmail.com
 */

#include "include/uart_queue.h"

ssize_t read_uart_queue(void *uart_queue, void* buffer, size_t size) {
    struct UART_Queue *queue = uart_queue;

    LOCK_UART_QUEUE(queue);

    // Loop for each byte we need to read.
    ssize_t i;
    for (i = 0; i < size; i++){

        // Attempt to pull the data from the queue, and break if the read fails.
        if (xQueueReceive(queue->read_queue, ((uint8_t*) buffer) + i, queue->transfer_timeout) != pdPASS) {
            break;
        }
    }

    UNLOCK_UART_QUEUE(queue);

    return i;
}

ssize_t write_uart_queue(void *uart_queue, void* buffer, size_t size) {
    struct UART_Queue *queue = uart_queue;

    LOCK_UART_QUEUE(queue);

    ROM_UARTIntEnable(queue->hardware_base_address, UART_INT_TX);

    // Loop for each byte we need to read.
    ssize_t i;
    for (i = 0; i < size; i++){

        // Attempt to push the data to the queue, and break if the write fails.
        if (xQueueSend(queue->write_queue, (uint8_t *)buffer + i, queue->transfer_timeout) != pdPASS) {
            break;
        }
    }

    UNLOCK_UART_QUEUE(queue);

    fill_tx_buffer(queue);

    return i;
}

void empty_rx_buffer(struct UART_Queue *queue) {

    BaseType_t higher_priority_task_woken = pdFALSE;
    uint8_t c;

    // Disable interrupts so that we don't get interrupted.
    ROM_IntDisable(queue->hardware_interrupt_address);


    // Loop while there are chars in the input buffer to read.
    while (ROM_UARTCharsAvail(queue->hardware_base_address)) {

        // Move a char from the hardware receive buffer to the read queue.
        c = (uint8_t) ROM_UARTCharGetNonBlocking(queue->hardware_base_address);
        xQueueSendToBackFromISR(queue->read_queue, &c, &higher_priority_task_woken);
    }

    portYIELD_FROM_ISR( higher_priority_task_woken );

    // Re-enable interrupts for the UART.
    ROM_IntEnable(queue->hardware_interrupt_address);
}

void fill_tx_buffer(struct UART_Queue *queue) {

    BaseType_t higher_priority_task_woken = pdFALSE;
    uint8_t data;

    // Disable interrupts so we don't get interrupted.
    ROM_IntDisable(queue->hardware_interrupt_address);

    // Loop while there is data that can be put into the transmit buffer.
    while (ROM_UARTSpaceAvail(queue->hardware_base_address) && (xQueueReceiveFromISR(queue->write_queue, &data, &higher_priority_task_woken) == pdPASS)) {
        // Move a single char from the write queue to the hardware transmit buffer.

        ROM_UARTCharPutNonBlocking(queue->hardware_base_address, data);
    }

    if (xQueueIsQueueEmptyFromISR(queue->write_queue) != pdFALSE) {

        ROM_UARTIntDisable(queue->hardware_base_address, UART_INT_TX);
        #ifdef DEBUG
        //UARTprintf("bedug\n");
        #endif

    } else {

        ROM_UARTIntEnable(queue->hardware_base_address, UART_INT_TX);
        #ifdef DEBUG
        //UARTprintf("debug\n");
        #endif
    }

    //portYIELD_FROM_ISR(higher_priority_task_woken);

    //Re-enable interrupts for the UART.
    ROM_IntEnable(queue->hardware_interrupt_address);
}
