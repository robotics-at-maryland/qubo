#ifndef _I2CTASK_H_
#define _I2CTASK_H_

//*****************************************************************************
//
// Number of I2C data packets to send.
//
//*****************************************************************************
#define NUM_I2C_DATA 3

//*****************************************************************************
//
// Set the address for slave module. This is a 7-bit address sent in the
// following format:
//                      [A6:A5:A4:A3:A2:A1:A0:RS]
//
// A zero in the "RS" position of the first byte means that the master
// transmits (sends) data to the selected slave, and a one in this position
// means that the master receives data from the slave.
//
//*****************************************************************************
#define SLAVE_ADDRESS 0x3C


//*****************************************************************************
//
// Configure the I2C0 master and slave and connect them using loopback mode.
//
//*****************************************************************************
int init_i2ctask(void);

void i2ctask(void* params);

#endif
