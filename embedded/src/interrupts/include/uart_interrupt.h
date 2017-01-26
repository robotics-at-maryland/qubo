/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _UART_INTERRUPT_H_
#define _UART_INTERRUPT_H_

#include "include/read_uart_queue.h"

QueueHandle_t read_uart_queue;

extern void UARTIntHandler(void);

#endif
