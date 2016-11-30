#ifndef _PROCESSINFO_H_
#define _PROCESSINFO_H_

#include "include/messages.h"

// How to interpret the messages
#define OPT_LED 0
#define OPT_PRINT 1

void process_uart_task(void* params);

#endif
