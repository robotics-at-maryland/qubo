#ifndef DEVICE_LOCKS_H
#define DEVICE_LOCKS_H

#include <FreeRTOS.h>
#include <semphr.h>

/*
  Mutexes and other semaphore type objects will be defined here.

  These variables will be allocated in main either in its stack or heap space
 */

extern SemaphoreHandle_t uart_mutex;

#endif
