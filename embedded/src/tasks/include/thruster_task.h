#ifndef _THRUSTER_TEST_H_
#define _THRUSTER_TEST_H_

#include <FreeRTOS.h>
#include <task.h>

#include "lib/include/bme280.h"

bool esc_test_init();

static void esc_test_task(void *params);

#endif
