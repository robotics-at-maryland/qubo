/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/*
  Define the globals that are used by both the i2c lib and i2c interrupt
*/

#ifndef _I2C0_GLOBALS_H_
#define _I2C0_GLOBALS_H_


// ****************************************************************************
// Variables
// ****************************************************************************

extern volatile uint32_t *i2c0_address;

// Buffer from i2c stored in this buffer
extern volatile uint8_t **i2c0_read_buffer;

// Stuff we want to write to i2c
extern volatile uint8_t **i2c0_write_buffer;

// How much bytes to read from i2c
extern volatile uint32_t *i2c0_read_count;

// How much bytes to write to i2c
extern volatile uint32_t *i2c0_write_count;

extern volatile uint16_t *i2c0_int_state;

#endif
