/* Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#ifndef _TIQU_H_
#define _TIQU_H_

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

#include "lib/include/uart_queue.h"
#include "lib/include/rgb.h"

#include "include/intertask_messages.h"
// #include "include/task_handles.h"
// #include "include/task_queues.h"

extern struct Depth_Status depth_status;
#define ERROR_FLAG 0x01
#define TRANSACTION_FLAG 0x02

/**
 * Creates the task used to communicate across the UART using Qubobus
 * data is transmitted through queues between the other tasks, and then
 * sent across the bus here
 * @return  0 on success
 */
bool tiqu_task_init(void);

/**
 * main qubobus task
 * @param params parameters handed to this task by FreeRTOS, we don't care
 */
static void tiqu_task(void *params);

#endif
