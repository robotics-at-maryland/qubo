/* Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#ifndef _TASK_QUEUES_H_
#define _TASK_QUEUES_H_

// FreeRTOS
#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>
#include <semphr.h>

// Tiva
#include <stdbool.h>
#include <stdint.h>
#include <inc/hw_memmap.h>
#include <driverlib/rom.h>
#include <driverlib/uart.h>

// Qubobus
#include "qubobus.h"
#include "io.h"

#define TASK_QUEUE_LENGTH 3

#define DECLARE_TASK_QUEUES QueueHandle_t embedded_queue, thruster_queue	/*, other_queue... */

#define INIT_TASK_QUEUES() do {											\
		embedded_queue = xQueueCreate(TASK_QUEUE_LENGTH, sizeof(QMsg)); \
		thruster_queue = xQueueCreate(TASK_QUEUE_LENGTH, sizeof(QMsg));	\
	} while (0)


typedef struct _QMsg{
	Transaction* transaction;
	Error* error;
	void* payload;
} QMsg;

extern QueueHandle_t embedded_queue;
extern QueueHandle_t thruster_queue;
#endif
