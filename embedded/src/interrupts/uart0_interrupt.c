/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com

   Greg Harris
   gharris1727@gmail.com
*/

#include <stdint.h>
#include <stdbool.h>
#include <inc/hw_nvic.h>
#include <inc/hw_types.h>
#include <inc/hw_memmap.h>
#include <driverlib/rom.h>
#include <driverlib/uart.h>

#include "interrupts/include/uart0_interrupt.h"
#include "lib/include/uart_queue.h"

// Handle an interrupt triggered for UART0
void UART0IntHandler(void) {
    volatile struct UART_Queue *queue = &uart0_queue;

	uint32_t status = ROM_UARTIntStatus(queue->hardware_base_address, true);

	if( status & (UART_INT_RX | UART_INT_RT) ){
        empty_rx_buffer(queue);
	}

	if( status & UART_INT_TX ){
        fill_tx_buffer(queue);
	}

    ROM_UARTIntClear(queue->hardware_base_address, ( UART_INT_TX | UART_INT_RX | UART_INT_RT));

}
