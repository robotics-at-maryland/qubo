/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/query_i2c.h"

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
      i2c_buffer = i2c0_buffer;
      i2c_count = i2c0_count;
      i2c_int_state = i2c0_int_state;
      break;
    }
  case I2C1_BASE:
    {
      i2c_mutex = &i2c1_mutex;
      i2c_address = i2c1_address;
      i2c_buffer = i2c1_buffer;
      i2c_count = i2c1_count;
      i2c_int_state = i2c1_int_state;
      break;
    }
  case I2C2_BASE:
    {
      i2c_mutex = &i2c2_mutex;
      i2c_address = i2c2_address;
      i2c_buffer = i2c2_buffer;
      i2c_count = i2c2_count;
      i2c_int_state = i2c2_int_state;
      break;
    }
  case I2C3_BASE:
    {
      i2c_mutex = &i2c3_mutex;
      i2c_address = i2c3_address;
      i2c_buffer = i2c3_buffer;
      i2c_count = i2c3_count;
      i2c_int_state = i2c3_int_state;
      break;
    }
  }
}

void writeI2C(uint32_t device, uint8_t addr, uint8_t *data, uint32_t length, bool query) {

  assign_vars(device);

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(*i2c_mutex, 0) == pdFALSE) {
    #ifdef DEBUG
    UARTprintf("Semaphore busy\n");
    #endif
    taskYIELD();
  }


  //
  // Save the data i2c0_buffer to be written.
  //
  *i2c_buffer = data;
  *i2c_count = length;
  *i2c_address = addr;

  #ifdef DEBUG
  UARTprintf("Writing %d bytes to %x\nWriting %x\n", *i2c_count, *i2c_address, **i2c0_buffer);
  #endif

  // Set the next state of the callback state machine based on the number of
  // bytes to write.
  if(length != 1 ) {
    *i2c_int_state = STATE_WRITE_NEXT;
  }
  else {
    *i2c_int_state = STATE_WRITE_FINAL;
  }


  // Set the slave i2c0_address and setup for a transmit operation.
  // Tiva shifts the address left, we need to offset it
  ROM_I2CMasterSlaveAddrSet(device, (*i2c_address)>>1, false);
  #ifdef DEBUG
  UARTprintf("Set slave addr to %x\nNext state is %d", *i2c_address, *i2c_int_state);
  #endif

  // Wait until the SoftI2C callback state machine is idle.
  while(*i2c_int_state != STATE_IDLE) {}

  #ifdef DEBUG
  UARTprintf("Finished interrupt\n");
  #endif

  // Only give it back if its an individual write
  if ( !query )
    xSemaphoreGive(*i2c_mutex);
}


void readI2C(uint32_t device, uint8_t addr, uint8_t *data, uint32_t length, bool query) {

  assign_vars(device);

  // Only take this if it's an individual read
  if ( !query ) {
    // If the i2c bus is busy, yield task and then try again
    while (xSemaphoreTake(i2c0_mutex, 0) == pdFALSE) {
      taskYIELD();
    }
  }

  // Save the data i2c0_buffer to be read.
  *i2c_buffer = data;
  *i2c_count = length;
  *i2c_address = addr;
  // Set the next state of the callback state machine based on the number of
  // bytes to read.
  if(length == 1)
    {
      *i2c_int_state = STATE_READ_ONE;
    }
  else
    {
      *i2c_int_state = STATE_READ_FIRST;
    }
  // Wait until the state machine is idle.
  while(*i2c_int_state != STATE_IDLE) {}

  xSemaphoreGive(*i2c_mutex);
}

// Will perform a write, then a read after
void queryI2C(uint32_t device, uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length) {

  writeI2C(device, addr, write_data, write_length, true);
  readI2C(device, addr, read_data, read_length, true);
}

