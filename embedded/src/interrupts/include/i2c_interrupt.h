/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _I2C_INTERRUPT_H_
#define _I2C_INTERRUPT_H_

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

#include "include/i2c_globals.h"

uint32_t address;

uint8_t *buffer = 0;

uint32_t count = 0;

volatile uint16_t int_state = STATE_IDLE;

void I2CIntHandler(void);

#endif
