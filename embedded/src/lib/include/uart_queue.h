/*
 * Greg Harris
 * R@M 2017
 * gharris1727@gmail.com
 */

/* Standard Library */
#include <stdint.h>
#include <unistd.h>

/* FreeRTOS */
#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>
#include <semphr.h>

/* TivaC Libraries */
#include <stdbool.h>
#include <driverlib/rom.h>
#include <driverlib/uart.h>

#ifndef UART_QUEUE_H
#define UART_QUEUE_H

/*
 * Type defining a collection of state for a uart connection.
 */
struct UART_Queue {

    /*
     * Hardware address values
     */
    uint32_t hardware_interrupt_address;
    uint32_t hardware_base_address;

    /*
     * Queues of data to send and receive.
     */
    volatile QueueHandle_t read_queue;
    volatile QueueHandle_t write_queue;

    /*
     * Lock to keep modifications of the queue from tasks atomic.
     */
    volatile SemaphoreHandle_t lock;

    /*
     * Timeout for transfers on the bus before failing to read.
     */
    TickType_t transfer_timeout;
};

/*
 * Macros for interacting with the uart state.
 */
#define INIT_UART_QUEUE(queue_v, read_buffer_size_v, write_buffer_size_v, hardware_interrupt_address_v, hardware_base_address_v, transfer_timeout_v) do { \
    (queue_v).hardware_interrupt_address = hardware_interrupt_address_v; \
    (queue_v).hardware_base_address = hardware_base_address_v; \
    (queue_v).transfer_timeout = transfer_timeout_v; \
    (queue_v).read_queue = xQueueCreate(read_buffer_size_v, sizeof(uint8_t)); \
    (queue_v).write_queue = xQueueCreate(write_buffer_size_v, sizeof(uint8_t)); \
    (queue_v).lock = xSemaphoreCreateMutex(); \
} while (0)

#define LOCK_UART_QUEUE(queue_p) xSemaphoreTake((queue_p)->lock, portMAX_DELAY)
#define UNLOCK_UART_QUEUE(queue_p) xSemaphoreGive((queue_p)->lock)

ssize_t read_uart_queue(void *uart_queue, void* buffer, size_t size);
ssize_t write_uart_queue(void *uart_queue, void* buffer, size_t size);

void empty_rx_buffer(struct UART_Queue *queue);
void fill_tx_buffer(struct UART_Queue *queue);

#endif
