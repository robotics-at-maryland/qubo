/* Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#ifndef _QUBOBUS_TEST_H
#define _QUBOBUS_TEST_H

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

#include "include/task_queues.h"
#include "include/task_handles.h"

#include "lib/include/rgb.h"

bool qubobus_test_init(void);

ssize_t qubobus_test_task(void* params);
#endif
