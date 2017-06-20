/*
 * Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#ifndef _TASK_HANDLES_H
#define _TASK_HANDLES_H

#include <FreeRTOS.h>
#include <task.h>

/*
 * This is a collection of Task handles, so that other files can  include
 * this and gain access to the handles of the other tasks on the system
 */

extern TaskHandle_t* qubobus_test_handle;

#define INIT_TASK_HANDLES() do {				\
		TaskHandle_t* qubobus_test_handle;		\
	} while (0)

#endif
