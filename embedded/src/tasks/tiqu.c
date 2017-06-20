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


// Handles requests received from the bus
static uint8_t handle_request(IO_State *state, Message *message, const uint8_t* buffer){

	QMsg q_msg;

	// Get the data from the task
	switch (message->header.message_type){

	case M_ID_EMBEDDED_STATUS: {
		// Notify using the ID of the request, so tasks know what to do
		xTaskNotify(qubobus_test_handle, M_ID_EMBEDDED_STATUS, eSetValueWithOverwrite);
		if(xQueueReceive(embedded_queue, (void*)&q_msg, ((struct UART_Queue*) state->io_host)->transfer_timeout ) != pdTRUE) {
			return -1;

		}
		break;
	}

	default:
		return -1;
	}

	// Now write it
	Message response;
	if ( q_msg.transaction != NULL){
		response = create_response(q_msg.transaction, q_msg.payload);
	} else if ( q_msg.error != NULL) {
		response = create_error(q_msg.error, q_msg.payload);
	}

	if ( write_message( state, &response)){
		return -1;
	}
	// Everything worked
	pvPortFree(q_msg.payload);
	q_msg.payload = NULL;
	return 0;
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
				if (handle_request(&state, &message, buffer)){
					goto reconnect;
				}
  				break;
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

