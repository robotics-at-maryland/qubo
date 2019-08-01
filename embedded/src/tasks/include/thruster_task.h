/* Jeremy Weed
 * R@M 2017
 * jweed262@umd.edu
 */

#ifndef _THRUSTER_TEST_H_
#define _THRUSTER_TEST_H_

// FreeRTOS
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>
#include <message_buffer.h>

#include "include/intertask_messages.h"

// Hardware libraries
#include "lib/include/pca9685.h"
#include "lib/include/rgb.h"

// Qubobus
#include "qubobus.h"

// which bus on the Tiva the PCA9685 is connected to
#define THRUSTER_I2C_BUS I2C0_BASE
// I2C address of the PCA9685
#define PCA_ADDR 0b1111111
// Frequency of the PWM on the PCA9685.  It's a float.
#define PWM_FREQ 60.0
// Maximum time, in microseconds, to set the PWM to
#define MAX_PULSE 2000
// Minimum time, in microseconds, to set the PWM to
#define MIN_PULSE 800
// Convert a float [-1, 1] into a value the PCA understands
#define THRUSTER_SCALE(x) ((x)*(MAX_PULSE-MIN_PULSE)/2 + MIN_PULSE)/(1E6/PWM_FREQ)*4096
bool thruster_task_init();

static void thruster_task(void *params);

#endif
