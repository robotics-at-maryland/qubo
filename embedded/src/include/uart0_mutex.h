/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _UART0_MUTEX_H_
#define _UART0_MUTEX_H_

#include <FreeRTOS.h>
#include <semphr.h>

extern SemaphoreHandle_t uart0_write_mutex;

extern SemaphoreHandle_t uart0_read_mutex;

#endif
