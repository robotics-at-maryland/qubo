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


// extern MessageBufferHandle_t thruster_message_buffer;

// These are instantiated in "main.c"
extern QueueHandle_t thruster_message_buffer;
extern QueueHandle_t ads_message_buffer;

// Macro to use in "main.c" so that everything gets created in one line
// #define INIT_MESSAGE_BUFFERS() do {                                  \
//		thruster_message_buffer = xMessageBufferCreate(sizeof(struct Thruster_Set)); \
//	} while (0)


#endif
