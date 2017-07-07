/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _READ_UART1_QUEUE_H_
#define _READ_UART1_QUEUE_H_

#include <FreeRTOS.h>
#include <queue.h>

#define READ_UART1_Q_SIZE 64

extern volatile QueueHandle_t read_uart1_queue;

#endif
