/*
  A message will be a parameter that is passed into a task. Tasks that communicate with other
  tasks will need a message parameter that gives it some FreeRTOS data structure which will
  be allocated in main.c

  This is so source code in tasks is clearer to read.
 */

typedef QueueHandle_t read_uart_msg;

