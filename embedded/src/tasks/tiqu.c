/*
 * Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#include "tasks/include/tiqu.h"

extern struct UART_Queue uart0_queue;
static char buffer[QUBOBUS_MAX_PAYLOAD_LENGTH];
// This is where a message received from a queue will be put.
void* payload;

bool tiqu_task_init(void){
	if ( xTaskCreate(tiqu_task, (const portCHAR *) "Tiva Qubobus", 1024, NULL,
					 tskIDLE_PRIORITY + 2, NULL) != pdTRUE) {
		return true;
	}
	return false;
}


// Handles requests received from the bus
static uint8_t handle_request(IO_State *state, Message *message, const uint8_t* buffer){

	Transaction transaction;
	Error error;
	Message response;
	uint8_t flag = 0;

	// Get the data from the task
	// Start with the highest message id, and use else-if to check like a switch-case

	// Maximum offset value.  Nothing should be higher than this
	if (message->header.message_id >= M_ID_OFFSET_MAX) {

		// If the ID is higher than max, something is wrong
		return -1;
	}

	// Debug offset, mostly things concerned with logging
	else if (message->header.message_id >= M_ID_OFFSET_DEBUG) {

		switch (message->header.message_id) {
		case M_ID_DEBUG_LOG_READ: {
			break;
		}
		case M_ID_DEBUG_LOG_ENABLE: {
			break;
		}
		case M_ID_DEBUG_LOG_DISABLE: {
			break;
		}
		default:
			break;
		}
	}

	// Depth offset, used for reading/setting the depth monitor
	else if (message->header.message_id >= M_ID_OFFSET_DEPTH) {

		switch (message->header.message_id) {

		case M_ID_DEPTH_STATUS: {
			break;
		}
		case M_ID_DEPTH_MONITOR_ENABLE: {
			break;
		}
		case M_ID_DEPTH_MONITOR_DISABLE: {
			break;
		}
		case M_ID_DEPTH_MONITOR_SET_CONFIG: {
			break;
		}
		case M_ID_DEPTH_MONITOR_GET_CONFIG: {
			break;
		}
		}
	}

	// Pneumatics offset.  Currently not implemented in hardware
	else if (message->header.message_id >= M_ID_OFFSET_PNEUMATICS) {

		// There's only one thing to break out here
	}

	// Thruster offset, For setting/getting thruster values
	else if (message->header.message_id >= M_ID_OFFSET_THRUSTER) {

		switch (message->header.message_id) {

		case M_ID_THRUSTER_SET: {

			/* create the message */
			struct Thruster_Set* thruster_set = (struct Thruster_Set*) message->payload;

			/* send it to the task*/
			if (
				/* xMessageBufferSend(thruster_message_buffer, */
				/*				   (void*) thruster_set, */
				/*					sizeof(thruster_set), */
				/*					pdMS_TO_TICKS(10)) == 0 */
				xQueueSend(thruster_message_buffer,
						   (void*) thruster_set,
								   pdMS_TO_TICKS(10)) == 0
				) {
				error = eThrusterUnreachable;
				flag = ERROR_FLAG;
			}
			transaction = tThrusterSet;
			flag = TRANSACTION_FLAG;
		}
		}
	}

	// Power offset.  Used to set the monitor/get status
	else if (message->header.message_id >= M_ID_OFFSET_POWER) {

		switch (message->header.message_id) {
		case M_ID_POWER_STATUS: {
			break;
		}
		case M_ID_POWER_RAIL_ENABLE: {
			break;
		}
		case M_ID_POWER_RAIL_DISABLE: {
			break;
		}
		case M_ID_POWER_MONITOR_ENABLE: {
			break;
		}
		case M_ID_POWER_MONITOR_DISABLE: {
			break;
		}
		case M_ID_POWER_MONITOR_SET_CONFIG: {
			break;
		}
		case M_ID_POWER_MONITOR_GET_CONFIG: {
			break;
		}
		}
	}

	// Battery offset, used to enable monitor and get status
	else if (message->header.message_id >= M_ID_OFFSET_BATTERY) {

		switch (message->header.message_id) {
		case M_ID_BATTERY_STATUS: {

		}
		case M_ID_BATTERY_SHUTDOWN: {
			break;
		}
		case M_ID_BATTERY_MONITOR_ENABLE: {
			break;
		}
		case M_ID_BATTERY_MONITOR_DISABLE: {
			break;
		}
		case M_ID_BATTERY_MONITOR_SET_CONFIG: {
			break;
		}
		case M_ID_BATTERY_MONITOR_GET_CONFIG: {
			break;
		}
		}
	}

	// Safety offset.  Used for enabling/reading the safety
	else if (message->header.message_id >= M_ID_OFFSET_SAFETY) {

		switch (message->header.message_id) {
		case M_ID_SAFETY_STATUS: {
			break;
		}
		case M_ID_SAFETY_SET_SAFE: {
			break;
		}
		case M_ID_SAFETY_SET_UNSAFE: {
			break;
		}
		}
	}

	// Embedded offset.  Used for things involving the embedded system as a whole
	else if (message->header.message_id >= M_ID_OFFSET_EMBEDDED) {

		// Notify using the ID of the request, so tasks know what to do
		/* xTaskNotify(qubobus_test_handle, message->header.message_id, eSetValueWithOverwrite); */
		/* if(xQueueReceive(embedded_queue, (void*)&q_msg, */
		/*				 ((struct UART_Queue*)state->io_host)->transfer_timeout ) != pdPASS) { */
		return -1;

	}

	// Core offset.  Used for things that involve the qubobus core protocol
	else if (message->header.message_id >= M_ID_OFFSET_CORE) {
		// There should only be errors from this thing
		return -1;
	}

	// Minimum offset, we shouldn't see this
	else if (message->header.message_id >= M_ID_OFFSET_MIN) {
		// If we get here, something is wrong.
		return -1;
	}
	// did we get something back, or was there an error?
	switch (flag) {
	case ERROR_FLAG: {
		response = create_error(&error, payload);
		break;
	}
	case TRANSACTION_FLAG: {
		response = create_response(&transaction, payload);
		break;
	}
	default: {
		// Bad message, send back the protocol error
		response = create_error(&eProtocol, &response);
		break;
	}
	}

	if ( write_message( state, &response)){
		blink_rgb(RED_LED, 1);
		return -1;
	}
	// Everything worked
	return 0;
}


static uint8_t handle_error(IO_State *state, Message *message, const uint8_t* buffer){
	switch ( message->header.message_id ) {
	case E_ID_CHECKSUM: {
		// When we get a checksum error, we re-transmit the message
		Message response;

		return 0;

	}
	default: {
		return -1;
	}
	}
}

static void tiqu_task(void *params){
	int error = 1;
	Message message;

	IO_State state = initialize(&uart0_queue, read_uart_queue, write_uart_queue, 1);

	for(;;){
		// This is where we jump to if something goes wrong on the bus
		reconnect:
		// wait for the bus to connect
		while( wait_connect( &state, buffer ))  {blink_rgb(RED_LED, 1);}
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
					blink_rgb(BLUE_LED, 1);
					goto reconnect;
				}
				break;
			}
			case MT_REQUEST: {

				if (handle_request(&state, &message, buffer)){
					goto reconnect;
				}
				break;
			}
			case MT_RESPONSE: {
				goto reconnect;
			}
			case MT_ERROR: {

				if ( handle_error ( &state, &message, buffer )){
					goto reconnect;
				}
				break;
			}
			default:
				// something is wrong, break comms and reconnect
				goto reconnect;
			}
		}
	}

}
