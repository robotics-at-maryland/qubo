/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _I2C0_INTERRUPT_H_
#define _I2C0_INTERRUPT_H_

// Change this to the i2c device to use
#define I2C_DEVICE I2C0_BASE

#include <stdbool.h>
#include <stdint.h>

#include <inc/hw_ints.h>
#include <inc/hw_memmap.h>
#include <driverlib/rom.h>
#include <driverlib/gpio.h>
#include <driverlib/interrupt.h>
#include <driverlib/pin_map.h>
#include <driverlib/sysctl.h>
#include <driverlib/i2c.h>

#include "interrupts/include/i2c_states.h"
#include "include/i2c0_globals.h"

#ifdef DEBUG
#include <utils/uartstdio.h>
#endif

volatile uint32_t *i2c0_address;

// Buffer from i2c stored in this buffer
volatile uint8_t **i2c0_read_buffer;

// Stuff we want to write to i2c
volatile uint8_t **i2c0_write_buffer;

// How much bytes to read from i2c
volatile uint32_t *i2c0_read_count;

// How much bytes to write to i2c
volatile uint32_t *i2c0_write_count;

volatile uint16_t *i2c0_int_state;

void I2C0IntHandler(void);

#endif
