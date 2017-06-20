/*
 * Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#include "tasks/include/qubobus_test.h"


bool qubobus_test_init(void){
	if (xTaskCreate(qubobus_test_task, (const portCHAR*) "Qubobus Test", 512, NULL,
					tskIDLE_PRIORITY + 1, qubobus_test_handle) != pdTRUE) {
		return true;
	}
	return false;
}

ssize_t qubobus_test_task(void* params){
	uint32_t notif_val;
	for (;;) {
		xTaskNotifyWait(0x0, 0xFFFF, &notif_val, portMAX_DELAY);
		QMsg msg;
		msg.transaction = &tEmbeddedStatus;
		msg.error = NULL;
		msg.payload = pvPortMalloc(msg.transaction->response);
		((struct Embedded_Status*) msg.payload)->uptime = xTaskGetTickCount() / (float)configTICK_RATE_HZ;
		// Don't know if  its possible to get this
		((struct Embedded_Status*) msg.payload)->mem_capacity = -1;
		xQueueSend(embedded_queue, &msg, 0);
	}
}
