/*
 * Kyle Montemayor
 * kmonte@umd.edu
 * R@M 2017
 */

#ifndef _BME280_TASK_H_
#define _BME280_TASK_H_

#include <FreeRTOS.h>
#include <task.h>

#include "lib/include/bme280.h"

bool bme280_task_init(void);

static void bme280_task_loop(void *params);

#endif
