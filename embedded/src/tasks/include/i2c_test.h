#ifndef _I2C_TEST_H_
#define _I2C_TEST_H_

#include <FreeRTOS.h>
#include <task.h>

#include "lib/include/bme280.h"

bool i2c_test_init();

static void i2c_test_task(void *params);

#endif
