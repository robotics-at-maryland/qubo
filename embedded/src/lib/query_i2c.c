/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/query_i2c.h"

void initI2C(void) {
  //
  // Enable the peripherals used by this example.
  //
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C0);
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);


  //
  // Configure the pin muxing for I2C0 functions on port B2 and B3.
  // This step is not necessary if your part does not support pin muxing.
  // TODO: change this to select the port/pin you are using.
  //
  ROM_GPIOPinConfigure(GPIO_PB2_I2C0SCL);
  ROM_GPIOPinConfigure(GPIO_PB3_I2C0SDA);

  // Select the I2C function for these pins.
  ROM_GPIOPinTypeI2CSCL(GPIO_PORTB_BASE, GPIO_PIN_2);
  ROM_GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_3);


  //
  // Initialize the I2C master.
  //
  ROM_I2CMasterInitExpClk(I2C_LIB_DEVICE, ROM_SysCtlClockGet(), false);

  //
  // Enable the I2C interrupt.
  //
  ROM_IntEnable(INT_I2C0);

  //
  // Enable the I2C master interrupt.
  //
  ROM_I2CMasterIntEnable(I2C_LIB_DEVICE);
}


void writeI2C(uint8_t addr, uint8_t *data, uint32_t length) {

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(i2c_mutex, 0) == pdFALSE) {
    taskYIELD();
  }

  //
  // Save the data buffer to be written.
  //
  buffer = data;
  count = length;
  address = addr;

  // Set the next state of the callback state machine based on the number of
  // bytes to write.
  if(count != 1 ) {
    int_state = STATE_WRITE_NEXT;
  }
  else {
    int_state = STATE_WRITE_FINAL;
  }

  // Set the slave address and setup for a transmit operation.
  ROM_I2CMasterSlaveAddrSet(I2C_LIB_DEVICE, address, false);

  // Wait until the SoftI2C callback state machine is idle.
  while(int_state != STATE_IDLE)
    {
    }
  xSemaphoreGive(i2c_mutex);
}


void readI2C(uint8_t addr, uint8_t *data, uint32_t length) {

  // If the i2c bus is busy, yield task and then try again
  while (xSemaphoreTake(i2c_mutex, 0) == pdFALSE) {
    taskYIELD();
  }
  // Save the data buffer to be read.
  buffer = data;
  count = length;
  address = addr;
  // Set the next state of the callback state machine based on the number of
  // bytes to read.
  if(length == 1)
    {
      int_state = STATE_READ_ONE;
    }
  else
    {
      int_state = STATE_READ_FIRST;
    }
  // Wait until the SoftI2C callback state machine is idle.
  while(int_state != STATE_IDLE)
    {
    }
  xSemaphoreGive(i2c_mutex);
}

// Will perform a write, then a read after
void queryI2C(uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length) {

  writeI2C(addr, write_data, write_length);
  readI2C(addr, read_data, read_length);
}

