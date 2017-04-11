/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _EXAMPLE_TEST_H_
#define _EXAMPLE_TEST_H_

#include "include/task_constants.h"
#include "lib/include/pca9685.h"

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

static void servo_test_task(void *params);

bool servo_test_init(void);

#endif
