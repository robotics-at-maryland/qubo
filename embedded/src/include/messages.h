/*
  A message will be a parameter that is passed into a task. Tasks that communicate with other
  tasks will need a message parameter that gives it some FreeRTOS data structure which will
  be allocated in main.c

  This is so source code in tasks is clearer to read.
 */

#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>

// The max size a message sent from computer to MCU can be
#define UART_BUFFER_SIZE 10

/**
  Queue of individual bytes from UART Interrupt to the task
*/
typedef QueueHandle_t read_uart_msg;

/**
   message_queue = pointer to queue of completed messages received from computer

   Semaphore pointers to tasks to unblock depending on the message
*/
typedef struct _process_uart_msg
{
  read_uart_msg *message_queue;
  SemaphoreHandle_t *insert_tasks_to_unblock;
} process_uart_msg;
