#include <FreeRTOS.h>
#include <queue.h>

#define READ_UART_Q_SIZE 64

extern QueueHandle_t read_uart_queue;
