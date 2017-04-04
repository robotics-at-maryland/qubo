/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/*
  Define the globals that are used by both the i2c lib and i2c interrupt
*/

#ifndef _I2C3_GLOBALS_H_
#define _I2C3_GLOBALS_H_


// ****************************************************************************
// Variables
// ****************************************************************************

extern uint32_t *i2c3_address;

// Buffer from i2c stored in this buffer
extern uint8_t **i2c3_read_buffer;

// Stuff we want to write to i2c
extern uint8_t **i2c3_write_buffer;

// How much bytes to read from i2c
extern uint32_t *i2c3_read_count;

// How much bytes to write to i2c
extern uint32_t *i2c3_write_count;

extern uint16_t *i2c3_int_state;


#endif
