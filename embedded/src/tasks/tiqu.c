/*
 * Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#include "tasks/include/tiqu.h"

extern struct UART_Queue uart0_queue;
static char buffer[QUBOBUS_MAX_PAYLOAD_LENGTH];
// This is where a message received from a queue will be put.
static QMsg q_msg = {.transaction = NULL, .error =  NULL, .payload = NULL};

bool tiqu_task_init(void){
	if ( xTaskCreate(tiqu_task, (const portCHAR *) "Tiva Qubobus", 1024, NULL,
					 tskIDLE_PRIORITY + 2, NULL) != pdTRUE) {
		return true;
	}
	return false;
}


// Handles requests received from the bus
static uint8_t handle_request(IO_State *state, Message *message, const uint8_t* buffer){

	// Get the data from the task
	switch (message->header.message_id){

	case M_ID_EMBEDDED_STATUS: {

		// Notify using the ID of the request, so tasks know what to do
		xTaskNotify(qubobus_test_handle, M_ID_EMBEDDED_STATUS, eSetValueWithOverwrite);
		if(xQueueReceive(embedded_queue, (void*)&q_msg,
						 ((struct UART_Queue*)state->io_host)->transfer_timeout ) != pdPASS) {
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
	} else {
		// Something went wrong, just give up
		return -1;
	}

	if ( write_message( state, &response)){
		blink_rgb(RED_LED, 1);
		return -1;
	}
	// Everything worked
	return 0;
}

static void tiqu_task(void *params){
	int error = 1;
	Message message;

	IO_State state = initialize(&uart0_queue, read_uart_queue, write_uart_queue, 1);

	for(;;){
		// This is where we jump to if something goes wrong on the bus
		reconnect:
		// wait for the bus to connect
		while( wait_connect( &state, buffer )); /* {blink_rgb(RED_LED | GREEN_LED, 1);} */
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
				if ( write_message( &state, &message ) != 0){
					goto reconnect;
				}
				break;
			}
			case MT_REQUEST: {

				// we free here so that the previous message can hang around so we
				// can re-transmit it in case of a checksum error
				vPortFree(q_msg.payload);
				q_msg.payload = NULL;
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

