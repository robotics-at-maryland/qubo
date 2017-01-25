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


// The I2C Bus to use
#define I2C_LIB_DEVICE I2C0_BASE


//*****************************************************************************
// The states in the interrupt handler state machine.
//*****************************************************************************
#define STATE_IDLE              0
#define STATE_WRITE_NEXT        1
#define STATE_WRITE_FINAL       2
#define STATE_WAIT_ACK          3
#define STATE_SEND_ACK          4
#define STATE_READ_ONE          5
#define STATE_READ_FIRST        6
#define STATE_READ_NEXT         7
#define STATE_READ_FINAL        8
#define STATE_READ_WAIT         9

// ****************************************************************************
// Variables
// ****************************************************************************

// Address of slave
static uint32_t address;

// The variables that track the data to be transmitted or received.
static uint8_t *buffer = 0;
static uint32_t count = 0;

// The current state of the interrupt handler state machine.
static volatile uint16_t int_state = STATE_IDLE;

// Mutex on I2C bus, declared extern so it exists for all classes using this lib
SemaphoreHandle_t i2c_mutex;

// ***************************************************************************
// Functions
// ***************************************************************************


void initI2C(void);

void I2CIntHandler(void);

void writeI2C(uint8_t addr, uint8_t *data, uint32_t length);

void readI2C(uint8_t addr, uint8_t *data, uint32_t length);

// Will perform a write, then a read after
void queryI2C(uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length);

#endif
