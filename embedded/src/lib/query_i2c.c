/**
   Task that will read 
 */

#include "include/query_i2c.h"


void initI2C(void) {
  i2c_mutex = xSemaphoreCreateMutex();
}

void SoftI2CCallback(void)
{
    //
    // Clear the SoftI2C interrupt.
    //
    SoftI2CIntClear(&state);

    //
    // Determine what to do based on the current state.
    //
    switch(state)
    {
        //
        // The idle state.
        //
        case STATE_IDLE:
        {
          xSemaphoreGiveFromISR(i2c_mutex, NULL);
        }

        //
        // The state for the middle of a burst write.
        //
        case STATE_WRITE_NEXT:
        {
            //
            // Write the next data byte.
            //
            SoftI2CDataPut(&state, *buffer++);
            size--;

            //
            // Continue the burst write.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_BURST_SEND_CONT);

            //
            // If there is one byte left, set the next state to the final write
            // state.
            //
            if(size == 1)
            {
                state = STATE_WRITE_FINAL;
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
            // Write the final data byte.
            //
            SoftI2CDataPut(&state, *buffer++);
            size--;

            //
            // Finish the burst write.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_BURST_SEND_FINISH);

            //
            // The next state is to wait for the burst write to complete.
            //
            state = STATE_SEND_ACK;

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
            if(SoftI2CErr(&state) == SOFTI2C_ERR_NONE)
            {
                //
                // Read the byte received.
                //
                SoftI2CDataGet(&state);

                //
                // There was no error, so the state machine is now idle.
                //
                state = STATE_IDLE;

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
            SoftI2CSlaveAddrSet(&state, SLAVE_ADDR, true);

            //
            // Perform a single byte read.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_SINGLE_RECEIVE);

            //
            // The next state is the wait for the ack.
            //
            state = STATE_WAIT_ACK;

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
            // Put the SoftI2C module into receive mode.
            //
            SoftI2CSlaveAddrSet(&state, SLAVE_ADDR, true);

            //
            // Perform a single byte read.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_SINGLE_RECEIVE);

            //
            // The next state is the wait for final read state.
            //
            state = STATE_READ_WAIT;

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
            // Put the SoftI2C module into receive mode.
            //
            SoftI2CSlaveAddrSet(&state, SLAVE_ADDR, true);

            //
            // Start the burst receive.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_BURST_RECEIVE_START);

            //
            // The next state is the middle of the burst read.
            //
            state = STATE_READ_NEXT;

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
            *buffer++ = SoftI2CDataGet(&state);
            size--;

            //
            // Continue the burst read.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_BURST_RECEIVE_CONT);

            //
            // If there are two characters left to be read, make the next
            // state be the end of burst read state.
            //
            if(size == 2)
            {
                state = STATE_READ_FINAL;
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
            *buffer++ = SoftI2CDataGet(&state);
            size--;

            //
            // Finish the burst read.
            //
            SoftI2CControl(&state, SOFTI2C_CMD_BURST_RECEIVE_FINISH);

            //
            // The next state is the wait for final read state.
            //
            state = STATE_READ_WAIT;

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
            *buffer++ = SoftI2CDataGet(&state);
            size--;

            //
            // The state machine is now idle.
            //
            state = STATE_IDLE;

            //
            // This state is done.
            //
            break;
        }
    }
}


boolean I2CWrite(uint8_t address, uint8_t *data, uint32_t length) {
  if ( xSemaphoreTake(i2c_mutex), 0) {

    //
    // Save the data buffer to be written.
    //
    buffer = data;
    size = length;

    //
    // Set the next state of the callback state machine based on the number of
    // bytes to write.
    //
    if(length != 1)
      {
        state = STATE_WRITE_NEXT;
      }
    else
      {
        state = STATE_WRITE_FINAL;
      }

    //
    // Set the slave address and setup for a transmit operation.
    //
    SoftI2CSlaveAddrSet(&state, SLAVE_ADDR, false);

    //
    // Write the first byte
    //
    SoftI2CDataPut(&state, *buffer);

    //
    // Start the burst cycle, writing the address as the first byte.
    //
    SoftI2CControl(&state, SOFTI2C_CMD_BURST_SEND_START);

    //
    // Wait until the SoftI2C callback state machine is idle.
    //
    while(state != STATE_IDLE)
      {
      }
    return true;
  }
  return false;
}


boolean I2CRead(uint8_t *data, uint32_t *length)
{
  //
  // Save the data buffer to be read.
  //
  buffer = data;
  size = length;

  //
  // Set the next state of the callback state machine based on the number of
  // bytes to read.
  //
  if(length == 1)
    {
      state = STATE_READ_ONE;
    }
  else
    {
      state = STATE_READ_FIRST;
    }

  //
  // Start with a dummy write to get the address set in the EEPROM.
  //
  SoftI2CSlaveAddrSet(&state, SLAVE_ADDR, false);

  //
  // Write the address to be written as the first data byte.
  //
  SoftI2CDataPut(&state, ui32Offset);

  //
  // Perform a single send, writing the address as the only byte.
  //
  SoftI2CControl(&state, SOFTI2C_CMD_SINGLE_SEND);

  //
  // Wait until the SoftI2C callback state machine is idle.
  //
  while(state != STATE_IDLE)
    {
    }
}
