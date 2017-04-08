/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Queue for the UART interrupt
#include "interrupts/include/uart0_interrupt.h"

// Interrupt for UART
void UART0IntHandler(void) {
	uint32_t status = ROM_UARTIntStatus(UART_DEVICE, true);

	#ifdef DEBUG
	//UARTprintf("interrupt triggered\n");
	#endif

	if( status & UART_INT_RX ){
		//just clear the read interrupt
		ROM_UARTIntClear(UART_DEVICE, UART_INT_RX);

		if( !ROM_UARTCharsAvail(UART_DEVICE) ){
			#ifdef DEBUG
			UARTprintf("read interrupt triggerd, but nothing to read\n");
			#endif
		}
		while( ROM_UARTCharsAvail(UART_DEVICE) ){
			//get the value + cast to uint8_t
			uint8_t c = (uint8_t) ROM_UARTCharGetNonBlocking(UART_DEVICE);
			//push to the queue
			#ifdef DEBUG
			//UARTprintf("read:%c\n", c);
			#endif
			if( xQueueSendToBackFromISR(read_uart0_queue, &c, NULL) != pdTRUE ){
			#ifdef DEBUG
			UARTprintf("error pushing to queue\n");
			#endif
			}
		}

	}
	if( status & UART_INT_RT ){
		ROM_UARTIntClear(UART_DEVICE, UART_INT_RT);
        //the receive timeout is triggered when there's something in the queue
        //that we didn't handle
		#ifdef DEBUG
		UARTprintf("recieve timeout\n");
		#endif
        //get the value + cast to uint8_t
        uint8_t c = (uint8_t) ROM_UARTCharGetNonBlocking(UART_DEVICE);
        //push to the queue
        #ifdef DEBUG
        //UARTprintf("read:%c\n", c);
        #endif
        if( xQueueSendToBackFromISR(read_uart0_queue, &c, NULL) != pdTRUE ){
        #ifdef DEBUG
        UARTprintf("error pushing to queue\n");
        #endif
        }
	}

	if( status & UART_INT_TX ){
		ROM_UARTIntClear(UART_DEVICE, UART_INT_TX);
		//push everything we need to write to the UART FIFO
		writeUART0();
		//writeUART1();
	}
	/*
	// Clear interrupt
	ROM_UARTIntClear(UART_DEVICE, status);

	if ( !ROM_UARTCharsAvail(UART_DEVICE) ) {
	// ERROR, Do something here
	}

	// Get one byte
	while( ROM_UARTCharsAvail(UART_DEVICE) ){
		#ifdef DEBUG
		UARTprintf("reading to queue");
		#endif
	// Tivaware casts the byte to a int32_t for some reason, cast back to save space
	uint8_t c = (uint8_t)(ROM_UARTCharGet(UART_DEVICE));

	//ROM_UARTCharPutNonBlocking(UART_DEVICE, c);
	// Push to the queue

	xQueueSendToBackFromISR(read_uart0_queue, &c, NULL);
	}
	*/
}
