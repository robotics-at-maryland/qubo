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

extern TaskHandle_t qubobus_test_handle;

#define DECLARE_TASK_HANDLES TaskHandle_t qubobus_test_handle /*, other_handle,...*/

#endif
