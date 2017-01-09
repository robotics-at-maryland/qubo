/**
   This file exposes these queues so that inner tasks to communicate to the computer,
   but having one task manage to communication ensures thread safety

   Declares the queue's inner tasks need to read or write to the
   outside computer.

   Reading from the computer is done by an Interrupt which writes to the read queue.

   When we want to send a message to the computer, we add it to the write queue
 */

#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>

extern QueueHandle_t write_uart;

extern QueueHandle_t read_uart;
