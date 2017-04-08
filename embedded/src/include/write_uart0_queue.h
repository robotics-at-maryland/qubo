/* Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#ifndef _WRITE_UART0_QUEUE_H
#define _WRITE_UART0_QUEUE_H

#include <FreeRTOS.h>
#include <queue.h>

#define WRITE_UART0_Q_SIZE 64

extern volatile QueueHandle_t write_uart0_queue;

#endif
