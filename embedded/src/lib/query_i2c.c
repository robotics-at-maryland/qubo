// THIS CODE WILL HAVE TO BE PORTED TO i2c.c HARDWARE LIBRARY

#include "include/query_i2c.h"

void initI2C(void) {
  i2c_mutex = xSemaphoreCreateMutex();
  //
  // For this example, PortB[3:2] are used for the SoftI2C pins.  GPIO port B
  // needs to be enabled so these pins can be used.
  // TODO: change this to whichever GPIO port(s) you are using.
  //
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);

  //
  // For this example, Timer0 is used for the SoftI2C time base.  This timer
  // needs to be enabled before it can be used.
  // TODO: change this to whichever timer you are using.
  //
  SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);

  //
  // Configure the appropriate pins to be I2C instead of GPIO.
  // TODO: change this to select the port/pin you are using.
  //
  GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_2 | GPIO_PIN_3);

  //
  // Initialize the SoftI2C module, including the assignment of GPIO pins.
  // TODO: change this to whichever GPIO pins you are using.
  //
  memset(&module_state, 0, sizeof(module_state));
  SoftI2CCallbackSet(&module_state, SoftI2CCallback);
  SoftI2CSCLGPIOSet(&module_state, GPIO_PORTB_BASE, GPIO_PIN_2);
  SoftI2CSDAGPIOSet(&module_state, GPIO_PORTB_BASE, GPIO_PIN_3);
  SoftI2CInit(&module_state);

  //
  // Enable the SoftI2C interrupt.
  //
  SoftI2CIntEnable(&module_state);

  //
  // Configure the timer to generate an interrupt at a rate of 40 KHz.  This
  // will result in a I2C rate of 10 KHz.
  // TODO: change this to whichever timer you are using.
  // TODO: change this to whichever I2C rate you require.
  //
  TimerConfigure(TIMER0_BASE, TIMER_CFG_PERIODIC);
  TimerLoadSet(TIMER0_BASE, TIMER_A, SysCtlClockGet() / 40000);
  TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
  TimerEnable(TIMER0_BASE, TIMER_A);

  //
  // Enable the timer interrupt.
  // TODO: change this to whichever timer interrupt you are using.
  //
  IntEnable(INT_TIMER0A);
}

void Timer0AIntHandler(void) {
  // Clear the timer interrupt.
  // TODO: change this to whichever timer you are using.
  TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

  // Call the SoftI2C tick function.
  SoftI2CTimerTick(&module_state);
}


void SoftI2CCallback(void) {
    //
    // Clear the SoftI2C interrupt.
    //
    SoftI2CIntClear(&module_state);

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
          xSemaphoreGiveFromISR(i2c_mutex, NULL);
        }

        // The state for the middle of a burst write.
        case STATE_WRITE_NEXT:
        {
            // Write the next data byte.
            SoftI2CDataPut(&module_state, *buffer++);
            size--;

            // Continue the burst write.
            SoftI2CControl(&module_state, SOFTI2C_CMD_BURST_SEND_CONT);

            // If there is one byte left, set the next state to the final write
            // state.
            if(size == 1)
            {
                int_state = STATE_WRITE_FINAL;
            }

            // This state is done.
            break;
        }

        // The state for the final write of a burst sequence.
        case STATE_WRITE_FINAL:
        {
            // Write the final data byte.
            SoftI2CDataPut(&module_state, *buffer++);
            size--;

            // Finish the burst write.
            SoftI2CControl(&module_state, SOFTI2C_CMD_BURST_SEND_FINISH);

            // The next state is to wait for the burst write to complete.
            int_state = STATE_SEND_ACK;

            // This state is done.
            break;
        }

        //
        // Wait for an ACK on the read after a write.
        //
        case STATE_WAIT_ACK:
        {
            // See if there was an error on the previously issued read.
            if(SoftI2CErr(&module_state) == SOFTI2C_ERR_NONE)
            {
                // Read the byte received.
                SoftI2CDataGet(&module_state);

                // There was no error, so the state machine is now idle.
                int_state = STATE_IDLE;

                //
                // This state is done.
                //
                break;
            }

            // Fall through to STATE_SEND_ACK.
        }

        // Send a read request, looking for the ACK to indicate that the write
        // is done.
        case STATE_SEND_ACK:
        {
            // Put the I2C master into receive mode.
            SoftI2CSlaveAddrSet(&module_state, address, true);

            // Perform a single byte read.
            SoftI2CControl(&module_state, SOFTI2C_CMD_SINGLE_RECEIVE);

            // The next state is the wait for the ack.
            int_state = STATE_WAIT_ACK;

            // This state is done.
            break;
        }

        // The state for a single byte read.
        case STATE_READ_ONE:
        {

            // Put the SoftI2C module into receive mode.
            SoftI2CSlaveAddrSet(&module_state, address, true);

            // Perform a single byte read.
            SoftI2CControl(&module_state, SOFTI2C_CMD_SINGLE_RECEIVE);

            // The next state is the wait for final read state.
            int_state = STATE_READ_WAIT;


            // This state is done.
            break;
        }

        // The state for the start of a burst read.
        case STATE_READ_FIRST:
        {
            // Put the SoftI2C module into receive mode.
            SoftI2CSlaveAddrSet(&module_state, address, true);

            // Start the burst receive.
            SoftI2CControl(&module_state, SOFTI2C_CMD_BURST_RECEIVE_START);

            // The next state is the middle of the burst read.
            int_state = STATE_READ_NEXT;

            // This state is done.
            break;
        }

        // The state for the middle of a burst read.
        case STATE_READ_NEXT:
        {

            // Read the received character.
            *buffer++ = SoftI2CDataGet(&module_state);
            size--;

            // Continue the burst read.
            SoftI2CControl(&module_state, SOFTI2C_CMD_BURST_RECEIVE_CONT);

            // If there are two characters left to be read, make the next
            // state be the end of burst read state.
            if(size == 2)
            {
                int_state = STATE_READ_FINAL;
            }

            // This state is done.
            break;
        }

        // The state for the end of a burst read.
        case STATE_READ_FINAL:
        {

            // Read the received character.
            *buffer++ = SoftI2CDataGet(&module_state);
            size--;

            // Finish the burst read.
            SoftI2CControl(&module_state, SOFTI2C_CMD_BURST_RECEIVE_FINISH);

            // The next state is the wait for final read state.
            int_state = STATE_READ_WAIT;

            // This state is done.
            break;
        }


        // This state is for the final read of a single or burst read.
        case STATE_READ_WAIT:
        {

            // Read the received character.
            *buffer++ = SoftI2CDataGet(&module_state);
            size--;

            // The state machine is now idle.
            int_state = STATE_IDLE;

            // This state is done.
            break;
        }
    }
}


bool I2CWrite(uint8_t addr, uint8_t *data, uint32_t length) {
  if ( xSemaphoreTake(i2c_mutex, 0) ) {

    //
    // Save the data buffer to be written.
    //
    buffer = data;
    // -1 because we're going to put the first byte in here
    size = length - 1;
    address = addr;

    // Set the next state of the callback state machine based on the number of
    // bytes to write.
    if(length != 1)
      {
        int_state = STATE_WRITE_NEXT;
      }
    else
      {
        int_state = STATE_WRITE_FINAL;
      }


    // Set the slave address and setup for a transmit operation.
    SoftI2CSlaveAddrSet(&module_state, address, false);

    // Write the first byte
    SoftI2CDataPut(&module_state, *buffer);


    // Start the burst cycle, writing the address as the first byte.
    SoftI2CControl(&module_state, SOFTI2C_CMD_BURST_SEND_START);

    // Wait until the SoftI2C callback state machine is idle.
    while(int_state != STATE_IDLE)
      {
      }
    return true;
  }
  return false;
}


bool I2CRead(uint8_t *data, uint32_t length) {

  // Save the data buffer to be read.
  buffer = data;
  size = length;

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


  // Start with a dummy write to get the address set in the EEPROM.
  SoftI2CSlaveAddrSet(&module_state, address, false);


  // Write the address to be written as the first data byte.
  SoftI2CDataPut(&module_state, *data);


  // Perform a single send, writing the address as the only byte.
  SoftI2CControl(&module_state, SOFTI2C_CMD_SINGLE_SEND);


  // Wait until the SoftI2C callback state machine is idle.
  while(int_state != STATE_IDLE)
    {
    }
}

// Will perform a write, then a read after
bool I2CQuery(uint8_t address, uint8_t *write_data, uint32_t write_length,
                 uint8_t *read_data, uint8_t *read_length) {
  return true;
}

