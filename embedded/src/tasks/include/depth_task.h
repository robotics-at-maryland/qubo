/* Nate Renegar 
 * R@M 2019
 * naterenegar@gmail.com 
 */

#ifndef DEPTH_TASK_H
#define DEPTH_TASK_H

// FreeRTOS
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>
#include <message_buffer.h>

#include "include/intertask_messages.h"

// Hardware Libraries
#include "lib/include/ms5837.h"

// Use the 3.3 V bus on the tiva
#define I2C_BUS I2C1_BASE
// Model of the MS5837 sensor
#define SENSOR_MODEL MS5837_30BA
bool depth_task_init();

static void depth_task(void *params);

#endif
