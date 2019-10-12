/* Jeremy Weed
 * R@M 2018
 * jweed262@umd.edu
 */

/*
 * Declarations for intertask message/stream buffers
 */

#ifndef _INTERTASK_MESSAGES_H_
#define _INTERTASK_MESSAGES_H_

#include <FreeRTOS.h>
#include <message_buffer.h>
#include <queue.h>
#include <semphr.h>

#include "qubobus.h"

extern QueueHandle_t thruster_message_buffer;

#endif
