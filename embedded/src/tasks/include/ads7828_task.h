/*
 * Nate Renegar
 * naterenegar@gmail.com 
 * R@M 2019 
 */

#ifndef _ADS7828_TASK_H_
#define _ADS7828_TASK_H_

#include <FreeRTOS.h>
#include <task.h>

#include "lib/include/ads7828.h"

bool ads7828_task_init(void);
static void ads7828_task_loop(void *params);

#endif 
