/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

/*
  Define the globals that are used by both the i2c lib and i2c interrupt
*/

#ifndef _I2C_GLOBALS_H_
#define _I2C_GLOBALS_H_

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
extern uint32_t address;

// The variables that track the data to be transmitted or received.
extern uint8_t *buffer;
extern uint32_t count;

// The current state of the interrupt handler state machine.
extern volatile uint16_t int_state;


#endif
