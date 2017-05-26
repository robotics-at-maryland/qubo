/*
 * Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#include "tasks/include/tiqu.h"

extern struct UART_Queue uart0_queue;
static char buffer[QUBOBUS_MAX_PAYLOAD_LENGTH];

bool tiqu_task_init(void){
	if ( xTaskCreate(tiqu_task, (const portCHAR *) "Tiva Qubobus", 1024, NULL,
					tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
		return true;
	}
	return false;
}

static void tiqu_task(void *params){
	int error = 1;
	Message message;

	IO_State state = initialize(&uart0_queue, read_uart_queue, write_uart_queue, 1);

	for(;;){
		// This is where we jump to if something goes wrong on the bus
		reconnect:
		blink_rgb(RED_LED, 1);
		// wait for the bus to connect
		while( wait_connect( &state, buffer ));
		blink_rgb(GREEN_LED, 1);
		for(;;){

			if( read_message( &state, &message, buffer ) != 0 ) break;

			switch ( message.header.message_type ){
			case MT_ANNOUNCE: {
				//looks like QSCU is trying to announce, we should reconnect
				goto reconnect;
			}
			case MT_PROTOCOL: {

			}
			case MT_KEEPALIVE: {
				// respond to the keepalive message
				message = create_keep_alive();
				if ( write_message( &state, &message )){
					goto reconnect;
				}
				blink_rgb(GREEN_LED | BLUE_LED, 1);
				break;
			}
			case MT_REQUEST: {

			}
			case MT_RESPONSE: {

			}
			case MT_ERROR: {

			}
			default:
				// something is wrong, break comms and reconnect
				goto reconnect;
			}
		}
	}

}
