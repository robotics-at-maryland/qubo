/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/**
   Library that can be used to send to I2C bus and receive data
 */

#ifndef _QUERY_I2C_H_
#define _QUERY_I2C_H_

#include "interrupts/include/i2c_states.h"

#include "include/i2c0_mutex.h"
#include "include/i2c1_mutex.h"
#include "include/i2c2_mutex.h"
#include "include/i2c3_mutex.h"
#include "include/i2c0_globals.h"
#include "include/i2c1_globals.h"
#include "include/i2c2_globals.h"
#include "include/i2c3_globals.h"

#include <FreeRTOS.h>
#include <semphr.h>
#include <task.h>

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <inc/hw_ints.h>
#include <inc/hw_memmap.h>
#include <driverlib/rom.h>
#include <driverlib/gpio.h>
#include <driverlib/interrupt.h>
#include <driverlib/pin_map.h>
#include <driverlib/sysctl.h>
#include <driverlib/i2c.h>

// Variables for the library call. These are set to the correct externs depending on the device param
// These are pointers because they'll be addressed to the externs already declared
SemaphoreHandle_t *i2c_mutex;

// Generic pointers that will be overwritten with specified globals
uint32_t *i2c_address;
uint8_t **i2c_read_buffer;
uint8_t **i2c_write_buffer;
uint32_t *i2c_read_count;
uint32_t *i2c_write_count;
uint16_t *i2c_int_state;

// Mutex on I2C bus, declared extern so it exists for all classes using this lib
SemaphoreHandle_t i2c0_mutex;
SemaphoreHandle_t i2c1_mutex;
SemaphoreHandle_t i2c2_mutex;
SemaphoreHandle_t i2c3_mutex;

volatile uint32_t *i2c0_address;
volatile uint8_t **i2c0_read_buffer;
volatile uint8_t **i2c0_write_buffer;
volatile uint32_t *i2c0_read_count;
volatile uint32_t *i2c0_write_count;
volatile uint16_t *i2c0_int_state;

volatile uint32_t *i2c1_address;
volatile uint8_t **i2c1_read_buffer;
volatile uint8_t **i2c1_write_buffer;
volatile uint32_t *i2c1_read_count;
volatile uint32_t *i2c1_write_count;
volatile uint16_t *i2c1_int_state;

volatile uint32_t *i2c2_address;
volatile uint8_t **i2c2_read_buffer;
volatile uint8_t **i2c2_write_buffer;
volatile uint32_t *i2c2_read_count;
volatile uint32_t *i2c2_write_count;
volatile uint16_t *i2c2_int_state;

volatile uint32_t *i2c3_address;
volatile uint8_t **i2c3_read_buffer;
volatile uint8_t **i2c3_write_buffer;
volatile uint32_t *i2c3_read_count;
volatile uint32_t *i2c3_write_count;
volatile uint16_t *i2c3_int_state;

// ***************************************************************************
// Functions
// ***************************************************************************

static void assign_vars(uint32_t device);

// Always set query to be false if you're specifically calling write or read.
// If query is true, the semaphore is taken at write, then given at read only
void writeI2C(uint32_t device, uint8_t addr, uint8_t *data, uint32_t length);

void readI2C(uint32_t device, uint8_t addr, uint8_t reg, uint8_t *data, uint32_t length);

#endif
