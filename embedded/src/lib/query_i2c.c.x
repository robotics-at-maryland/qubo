/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/query_i2c.h"


void writeI2C(uint32_t device, uint8_t addr, uint8_t *data, uint32_t length) {

  assign_vars(device);

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(*i2c_mutex, 0) == pdFALSE) {
    #ifdef DEBUG
    UARTprintf("Semaphore busy\n");
    #endif
    taskYIELD();
  }

  // Set address
  ROM_I2CMasterSlaveAddrSet(device, addr, false);

  // Put data, increment pointer after
  ROM_I2CMasterDataPut(device, (*data)++);

  if ( length == 1 ) {
    // Send byte
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_SINGLE_SEND);

    // Wait to finish transferring
    while(ROM_I2CMasterBusy(device));
  }

  else {
    // Initiate burst send
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_START);

    while(ROM_I2CMasterBusy(device));

    //
    for(uint32_t i = 0; i < (length - 1); i++) {
      // put data into fifo
      ROM_I2CMasterDataPut(device, *data);
      // Increment pointer
      data++;
      // Send data
      ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_CONT);
      // Wait till done transferring
      while(ROM_I2CMasterBusy(device));
    }

    // put last byte
    ROM_I2CMasterDataPut(device, *data);
    // Send last byte
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_FINISH);
    // wait till done transferring
    while(ROM_I2CMasterBusy(device));
  }

  // Give semaphore back
  xSemaphoreGive(*i2c_mutex);
}

void readI2C(uint32_t device, uint8_t addr, uint8_t reg, uint8_t *data, uint32_t length) {

  assign_vars(device);

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(i2c0_mutex, 0) == pdFALSE) {
    taskYIELD();
  }

  //specify that we are writing (a register address) to the
  //slave device
  ROM_I2CMasterSlaveAddrSet(device, addr, false);

  //specify register to be read
  ROM_I2CMasterDataPut(device, reg);

  //send control byte and register address byte to slave device
  ROM_I2CMasterControl(device, I2C_MASTER_CMD_SINGLE_SEND);

  //wait for MCU to finish transaction
  while(ROM_I2CMasterBusy(device));

  //specify that we are going to read from slave device
  ROM_I2CMasterSlaveAddrSet(device, addr, true);

  if ( length == 1 ) {
    // Read one byte
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_SINGLE_RECEIVE);

    // Wait to finish reading
    while(ROM_I2CMasterBusy(device));

    // Set the buffer to what was recieved
    *data = ROM_I2CMasterDataGet(device);
  }
  else {
    // Initiate burst read
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_RECEIVE_START);

    // Wait to finish reading
    while(ROM_I2CMasterBusy(device));

    for(uint32_t i = 0; i < (length - 1); i++) {
      // Set the buffer with recieved byte
      *data = ROM_I2CMasterDataGet(device);
      // Increment pointer
      data++;
      // Receive data
      ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_RECEIVE_CONT);
    }

    // Read last byte
    *data = ROM_I2CMasterDataGet(device);

    // Finish the read
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_RECEIVE_FINISH);

    while(ROM_I2CMasterBusy(device));
  }

  // Give back semaphore
  xSemaphoreGive(*i2c_mutex);
}

/*
  Private function whose only job is to assign the library pointers to the
  needed externs for interrupts
*/
static void assign_vars(uint32_t device) {
  switch(device) {
  case I2C0_BASE:
    {
      i2c_mutex = &i2c0_mutex;
      break;
    }
  case I2C1_BASE:
    {
      i2c_mutex = &i2c1_mutex;
      break;
    }
  case I2C2_BASE:
    {
      i2c_mutex = &i2c2_mutex;
      break;
    }
  case I2C3_BASE:
    {
      i2c_mutex = &i2c3_mutex;
      break;
    }
  }
}
