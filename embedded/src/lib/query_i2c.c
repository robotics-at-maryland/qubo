// THIS CODE WILL HAVE TO BE PORTED TO i2c.c HARDWARE LIBRARY

#include "include/query_i2c.h"

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

void I2CIntHandler(void) {
    //
    // Clear the I2C interrupt.
    //
    ROM_I2CMasterIntClear(I2C_LIB_DEVICE);

    //
    // Determine what to do based on the current state.
    //
    switch(int_state)
    {
        //
        // The idle state.
        //
        case STATE_IDLE:
        {
            //
            // There is nothing to be done.
            //
            break;
        }

        //
        // The state for the middle of a burst write.
        //
        case STATE_WRITE_NEXT:
        {
            //
            // Write the next byte to the data register.
            //
            ROM_I2CMasterDataPut(I2C_LIB_DEVICE, *buffer++);
            count--;

            //
            // Continue the burst write.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE, I2C_MASTER_CMD_BURST_SEND_CONT);

            //
            // If there is one byte left, set the next state to the final write
            // state.
            //
            if(count == 1)
            {
                int_state = STATE_WRITE_FINAL;
            }

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the final write of a burst sequence.
        //
        case STATE_WRITE_FINAL:
        {
            //
            // Write the final byte to the data register.
            //
            ROM_I2CMasterDataPut(I2C_LIB_DEVICE, *buffer++);
            count--;

            //
            // Finish the burst write.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE,
                             I2C_MASTER_CMD_BURST_SEND_FINISH);

            //
            // The next state is to wait for the burst write to complete.
            //
            int_state = STATE_SEND_ACK;

            //
            // This state is done.
            //
            break;
        }

        //
        // Wait for an ACK on the read after a write.
        //
        case STATE_WAIT_ACK:
        {
            //
            // See if there was an error on the previously issued read.
            //
            if(ROM_I2CMasterErr(I2C_LIB_DEVICE) == I2C_MASTER_ERR_NONE)
            {
                //
                // Read the byte received.
                //
                ROM_I2CMasterDataGet(I2C_LIB_DEVICE);

                //
                // There was no error, so the state machine is now idle.
                //
                int_state = STATE_IDLE;

                //
                // This state is done.
                //
                break;
            }

            //
            // Fall through to STATE_SEND_ACK.
            //
        }

        //
        // Send a read request, looking for the ACK to indicate that the write
        // is done.
        //
        case STATE_SEND_ACK:
        {
            //
            // Put the I2C master into receive mode.
            //
            ROM_I2CMasterSlaveAddrSet(I2C_LIB_DEVICE, address, true);

            //
            // Perform a single byte read.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE, I2C_MASTER_CMD_SINGLE_RECEIVE);

            //
            // The next state is the wait for the ack.
            //
            int_state = STATE_WAIT_ACK;

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for a single byte read.
        //
        case STATE_READ_ONE:
        {
            //
            // Put the I2C master into receive mode.
            //
            ROM_I2CMasterSlaveAddrSet(I2C_LIB_DEVICE, address, true);

            //
            // Perform a single byte read.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE, I2C_MASTER_CMD_SINGLE_RECEIVE);

            //
            // The next state is the wait for final read state.
            //
            int_state = STATE_READ_WAIT;

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the start of a burst read.
        //
        case STATE_READ_FIRST:
        {
            //
            // Put the I2C master into receive mode.
            //
            ROM_I2CMasterSlaveAddrSet(I2C_LIB_DEVICE, address, true);

            //
            // Start the burst receive.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE,
                             I2C_MASTER_CMD_BURST_RECEIVE_START);

            //
            // The next state is the middle of the burst read.
            //
            int_state = STATE_READ_NEXT;

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the middle of a burst read.
        //
        case STATE_READ_NEXT:
        {
            //
            // Read the received character.
            //
            *buffer++ = ROM_I2CMasterDataGet(I2C_LIB_DEVICE);
            count--;

            //
            // Continue the burst read.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE,
                             I2C_MASTER_CMD_BURST_RECEIVE_CONT);

            //
            // If there are two characters left to be read, make the next
            // state be the end of burst read state.
            //
            if(count == 2)
            {
                int_state = STATE_READ_FINAL;
            }

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the end of a burst read.
        //
        case STATE_READ_FINAL:
        {
            //
            // Read the received character.
            //
            *buffer++ = ROM_I2CMasterDataGet(I2C_LIB_DEVICE);
            count--;

            //
            // Finish the burst read.
            //
            ROM_I2CMasterControl(I2C_LIB_DEVICE,
                             I2C_MASTER_CMD_BURST_RECEIVE_FINISH);

            //
            // The next state is the wait for final read state.
            //
            int_state = STATE_READ_WAIT;

            //
            // This state is done.
            //
            break;
        }

        //
        // This state is for the final read of a single or burst read.
        //
        case STATE_READ_WAIT:
        {
            //
            // Read the received character.
            //
            *buffer++  = ROM_I2CMasterDataGet(I2C_LIB_DEVICE);
            count--;

            //
            // The state machine is now idle.
            //
            int_state = STATE_IDLE;

            //
            // This state is done.
            //
            break;
        }
    }
}


bool i2cWrite(uint8_t addr, uint8_t *data, uint32_t length) {
  if ( xSemaphoreTake(i2c_mutex, 0) ) {

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
    return true;
  }
  return false;
}


 bool i2cRead(uint8_t addr, uint8_t *data, uint32_t length) {

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
}

// Will perform a write, then a read after
bool i2cQuery(uint8_t addr, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t read_length) {
  bool write = i2cWrite(addr, write_data, write_length);
  bool read = i2cRead(addr, read_data, read_length);

  return write && read;
}

