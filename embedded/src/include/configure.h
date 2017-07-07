/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Initial setup of hardware

#ifndef _CONFIGURE_H_
#define _CONFIGURE_H_

// Tiva
#include <stdbool.h>
#include <stdint.h>
#include <inc/hw_memmap.h>
#include <inc/hw_types.h>
#include <inc/hw_ints.h>
#include <driverlib/interrupt.h>
#include <driverlib/gpio.h>
#include <driverlib/i2c.h>
#include <driverlib/pin_map.h>
#include <driverlib/rom.h>
#include <driverlib/sysctl.h>
#include <driverlib/uart.h>

void configureUART(void);

void configureGPIO(void);

void configureI2C(void);

#endif
