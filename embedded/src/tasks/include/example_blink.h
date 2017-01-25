/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _EXAMPLE_BLINK_H_
#define _EXAMPLE_BLINK_H_

#include "include/task_constants.h"
#include "lib/include/rgb.h"

// FreeRTOS
#include <FreeRTOS.h>
#include <queue.h>
#include <task.h>
#include <semphr.h>

// Tiva
#include <stdbool.h>
#include <stdint.h>
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <driverlib/gpio.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>
#include <utils/uartstdio.h>


#define RED_LED   GPIO_PIN_1
#define BLUE_LED  GPIO_PIN_2
#define GREEN_LED GPIO_PIN_3

static void example_blink_task(void *params);

bool example_blink_init(void);

#endif
