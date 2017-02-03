/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/query_i2c.h"

void writeI2C0(uint8_t addr, uint8_t *data, uint32_t length) {

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(i2c0_mutex, 0) == pdFALSE) {
    taskYIELD();
  }

  //
  // Save the data i2c0_buffer to be written.
  //
  i2c0_buffer = data;
  i2c0_count = length;
  i2c0_address = addr;

  // Set the next state of the callback state machine based on the number of
  // bytes to write.
  if(i2c0_count != 1 ) {
    i2c0_int_state = STATE_WRITE_NEXT;
  }
  else {
    i2c0_int_state = STATE_WRITE_FINAL;
  }

  // Set the slave i2c0_address and setup for a transmit operation.
  ROM_I2CMasterSlaveAddrSet(I2C0_BASE, i2c0_address, false);

  // Wait until the SoftI2C callback state machine is idle.
  while(i2c0_int_state != STATE_IDLE)
    {
    }
  xSemaphoreGive(i2c0_mutex);
}


void readI2C0(uint8_t addr, uint8_t *data, uint32_t length) {

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(i2c0_mutex, 0) == pdFALSE) {
    taskYIELD();
  }
  // Save the data i2c0_buffer to be read.
  i2c0_buffer = data;
  i2c0_count = length;
  i2c0_address = addr;
  // Set the next state of the callback state machine based on the number of
  // bytes to read.
  if(length == 1)
    {
      i2c0_int_state = STATE_READ_ONE;
    }
  else
    {
      i2c0_int_state = STATE_READ_FIRST;
    }
  // Wait until the state machine is idle.
  while(i2c0_int_state != STATE_IDLE)
    {
    }
  xSemaphoreGive(i2c0_mutex);
}

// Will perform a write, then a read after
void queryI2C0(uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length) {

  writeI2C0(addr, write_data, write_length);
  readI2C0(addr, read_data, read_length);
}

