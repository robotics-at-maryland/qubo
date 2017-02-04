/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/**
   Library that can be used to send to I2C bus and receive data
 */

#ifndef _QUERY_I2C_H_
#define _QUERY_I2C_H_

#include "include/i2c0_mutex.h"
#include "include/i2c0_globals.h"

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

uint32_t *i2c_address;
uint8_t **i2c_buffer;
uint32_t *i2c_count;
uint16_t *i2c_int_state;

// Mutex on I2C bus, declared extern so it exists for all classes using this lib
volatile SemaphoreHandle_t i2c0_mutex;

// Extern globals that are shared between interrupt
// I2C0_BASE:
volatile uint32_t i2c0_address;
volatile uint8_t *i2c0_buffer;
volatile uint32_t i2c0_count;
volatile uint16_t i2c0_int_state;

// ***************************************************************************
// Functions
// ***************************************************************************

static void assign_vars(uint32_t device);

void writeI2C(uint32_t device, uint8_t addr, uint8_t *data, uint32_t length);

void readI2C(uint32_t device, uint8_t addr, uint8_t *data, uint32_t length);

// Will perform a write, then a read after
void queryI2C(uint32_t device, uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length);

#endif
