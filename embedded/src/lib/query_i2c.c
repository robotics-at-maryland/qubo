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
    UARTprintf("query_i2c: Semaphore busy\n");
    #endif
    taskYIELD();
  }

  // Set address
  ROM_I2CMasterSlaveAddrSet(device, addr, false);

  // Put data for first send
  ROM_I2CMasterDataPut(device, *data);

  if ( length == 1 ) {
    // So the sceduler doesn't interrupt here
    taskENTER_CRITICAL();

    // Send byte
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_SINGLE_SEND);

    // Wait to finish transferring
    while(ROM_I2CMasterBusy(device)) {}

    taskEXIT_CRITICAL();
  }

  else if (length == 2) {
    // So the sceduler doesn't interrupt here
    taskENTER_CRITICAL();

    // Send first word
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_START);

    while(ROM_I2CMasterBusy(device));

    // Send second word
    ROM_I2CMasterDataPut(device, *(data + 1));
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_FINISH);

    while(ROM_I2CMasterBusy(device));

    taskEXIT_CRITICAL();
  }

  // This will use the interrupt
  else {
    *i2c_write_buffer = (data + 1);
    *i2c_write_count = (length - 1);
    *i2c_int_state = STATE_WRITE;
    #ifdef DEBUG
    UARTprintf("query_i2c: Setting STATE_WRITE\n");
    #endif

    // Initiate burst send
    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_START);

    #ifdef DEBUG
    UARTprintf("query_i2c: Init burst write\n");
    #endif

    /*
    while(ROM_I2CMasterBusy(device));

    for(uint32_t i = 0; i < (length - 1); i++) {
      // put data into fifo
      ROM_I2CMasterDataPut(device, *(data + 1));
      // Increment pointer
      data++;
      // Send data
      ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_SEND_CONT);
      // Wait till done transferring
      while(ROM_I2CMasterBusy(device));
    }
    */
  }
  // Don't return until done
  while(*i2c_int_state != STATE_IDLE) {
    vTaskDelay(250);
  }

  // Give semaphore back
  xSemaphoreGive(*i2c_mutex);
}

void readI2C(uint32_t device, uint8_t addr, uint8_t reg, uint8_t *data, uint32_t length) {

  assign_vars(device);

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(*i2c_mutex, 0) == pdFALSE) {
    taskYIELD();
  }

  //specify that we are writing (a register address) to the
  //slave device
  ROM_I2CMasterSlaveAddrSet(device, addr, false);

  //specify register to be read
  ROM_I2CMasterDataPut(device, reg);

  #ifdef DEBUG
  UARTprintf("query_i2c: Write register\n");
  #endif

  //send control byte and register address byte to slave device
  ROM_I2CMasterControl(device, I2C_MASTER_CMD_SINGLE_SEND);

  #ifdef DEBUG
  UARTprintf("query_i2c: wrote register\n");
  #endif

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
    *i2c_read_buffer = data;
    *i2c_read_count = length;
    *i2c_int_state = STATE_READ;
    #ifdef DEBUG
    UARTprintf("query_i2c: STATE_READ set\n");
    #endif

    ROM_I2CMasterControl(device, I2C_MASTER_CMD_BURST_RECEIVE_START);

    /*
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
    */
    #ifdef DEBUG
    UARTprintf("query_i2c: Test\n");
    #endif
  }

  while(*i2c_int_state != STATE_IDLE) {
    vTaskDelay(250);
  }

  #ifdef DEBUG
  UARTprintf("query_i2c: Not in idle anymore\n");
  #endif

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
      i2c_address = i2c0_address;
      i2c_read_buffer = i2c0_read_buffer;
      i2c_write_buffer = i2c0_write_buffer;
      i2c_read_count = i2c0_read_count;
      i2c_write_count = i2c0_write_count;
      i2c_int_state = i2c0_int_state;
      break;
    }
  case I2C1_BASE:
    {
      i2c_mutex = &i2c1_mutex;
      i2c_address = i2c1_address;
      i2c_read_buffer = i2c1_read_buffer;
      i2c_write_buffer = i2c1_write_buffer;
      i2c_read_count = i2c1_read_count;
      i2c_write_count = i2c1_write_count;
      i2c_int_state = i2c1_int_state;
      break;
    }
  case I2C2_BASE:
    {
      i2c_mutex = &i2c2_mutex;
      i2c_address = i2c2_address;
      i2c_read_buffer = i2c2_read_buffer;
      i2c_write_buffer = i2c2_write_buffer;
      i2c_read_count = i2c2_read_count;
      i2c_write_count = i2c2_write_count;
      i2c_int_state = i2c2_int_state;
      break;
    }
  case I2C3_BASE:
    {
      i2c_mutex = &i2c3_mutex;
      i2c_address = i2c3_address;
      i2c_read_buffer = i2c3_read_buffer;
      i2c_write_buffer = i2c3_write_buffer;
      i2c_read_count = i2c3_read_count;
      i2c_write_count = i2c3_write_count;
      i2c_int_state = i2c3_int_state;
      break;
    }
  }
}
