#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>

int main() {
	QueueHandle_t queue;
	queue = xQueueCreate(3,sizeof(int));
    return 0;
}

void vApplicationStackOverflowHook( TaskHandle_t pxTask, signed char *pcTaskName )
{
	while (1)
	{
					        /* my code. Prints stuff directly to the console*/
	}
}
