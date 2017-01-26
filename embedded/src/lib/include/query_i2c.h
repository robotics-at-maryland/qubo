/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/**
   Library that can be used to send to I2C bus and receive data
 */

#ifndef _SENDI2C_H_
#define _SENDI2C_H_

#include "include/i2c_mutex.h"
#include "include/i2c_globals.h"

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

// Mutex on I2C bus, declared extern so it exists for all classes using this lib
SemaphoreHandle_t i2c_mutex;

// Globals that the interrupt has access to as well
uint32_t address;

uint8_t *buffer;

uint32_t count;

volatile uint16_t int_state;

// ***************************************************************************
// Functions
// ***************************************************************************


void initI2C(void);

//void I2CIntHandler(void);

void writeI2C(uint8_t addr, uint8_t *data, uint32_t length);

void readI2C(uint8_t addr, uint8_t *data, uint32_t length);

// Will perform a write, then a read after
void queryI2C(uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length);

#endif
