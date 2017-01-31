/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _UART1_INTERRUPT_H_
#define _UART1_INTERRUPT_H_

// Change this to the device being used
#define UART_DEVICE UART1_BASE

#include "include/read_uart1_queue.h"

QueueHandle_t read_uart1_queue;

void UART1IntHandler(void);

#endif
