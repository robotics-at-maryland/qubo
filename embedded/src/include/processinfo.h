#ifndef _PROCESSINFO_H_
#define _PROCESSINFO_H_

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <semaphore.h>

#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>

// How to interpret the messages
#define OPT_LED 0
#define OPT_PRINT 1

typedef struct _processing_semaphores
{
  QueueHandle_T* input;
  SemaphoreHandle_t* led;
  SemaphoreHandle_t* print;
} processing_semaphores;

void processinfo_task(void* params);

#endif
