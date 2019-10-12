/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "interrupts/include/i2c3_interrupt.h"

void I2C3IntHandler(void) {
    // Clear the I2C interrupt.
    ROM_I2CMasterIntClear(I2C3_BASE);

    // Determine what to do based on the current state.
    switch(*i2c3_int_state)
    {
      //
      // The idle state.
      //
    case STATE_IDLE:
      {
        // Nothing happening
        #ifdef DEBUG
        UARTprintf("In idle .... \n");
        #endif
        break;
      }
    case STATE_WRITE:
      {
				#ifdef DEBUG
        UARTprintf("intrpt: In STATE_WRITE\n");
        UARTprintf("intrpt: buffer: %x\n", **i2c3_write_buffer);
        UARTprintf("intrpt: counter: %d\n", *i2c3_write_count);
				#endif
        // Put the current data in the buffer
        ROM_I2CMasterDataPut(I2C3_BASE, **i2c3_write_buffer); // Decrement the count pointer
        *i2c3_write_count = *i2c3_write_count - 1;
        // Point to the next byte
        *i2c3_write_buffer = *i2c3_write_buffer + 1;
        // Send the data
        ROM_I2CMasterControl(I2C3_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);

        // If on last byte, go to final stage
        if ( *i2c3_write_count <= 1 )
          *i2c3_int_state = STATE_WRITE_FINAL;

        break;
      }
    case STATE_WRITE_FINAL:
      {
        #ifdef DEBUG
        UARTprintf("intrpt: In STATE_WRITE_final\n");
        UARTprintf("intrpt: buffer: %x\n", **i2c3_write_buffer);
        UARTprintf("intrpt: counter: %d\n", *i2c3_write_count);
        #endif
        // Put data in buffer
        ROM_I2CMasterDataPut(I2C3_BASE, **i2c3_write_buffer);
        // Send last byte
        ROM_I2CMasterControl(I2C3_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

        // Since this is a query, next state is read
        *i2c3_int_state = STATE_IDLE;

        break;
      }
    case STATE_WRITE_QUERY:
      {
        // Put the current data in the buffer
        ROM_I2CMasterDataPut(I2C3_BASE, **i2c3_write_buffer);
        // Decrement the count pointer
        *i2c3_write_count = *i2c3_write_count - 1;
        // Point to the next byte
        *i2c3_write_buffer = *i2c3_write_buffer + 1;
        // Send the data
        ROM_I2CMasterDataPut(I2C3_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);

        // If on last byte, go to final stage
        if ( *i2c3_write_count <= 1 )
          *i2c3_int_state = STATE_WRITE_QUERY_FINAL;

        break;
      }
    case STATE_WRITE_QUERY_FINAL:
      {
        // Put data in buffer
        ROM_I2CMasterDataPut(I2C3_BASE, **i2c3_write_buffer);
        // Send last byte
        ROM_I2CMasterControl(I2C3_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

        // Put the master in recieve mode
        ROM_I2CMasterSlaveAddrSet(I2C3_BASE, *i2c3_address, true);
        ROM_I2CMasterControl(I2C3_BASE, I2C_MASTER_CMD_BURST_RECEIVE_START);

        // Since this is a query, next state is read
        *i2c3_int_state = STATE_READ;

        break;
      }
    case STATE_READ:
      {
        #ifdef DEBUG
        //UARTprintf("in STATE_READ\n");
        #endif
        // Save a byte
        **i2c3_read_buffer = ROM_I2CMasterDataGet(I2C3_BASE);
        // Increment the buffer
        *i2c3_read_buffer = *i2c3_read_buffer + 1;
        // Decrement the count
        *i2c3_read_count = *i2c3_read_count - 1;

        ROM_I2CMasterControl(I2C3_BASE, I2C_MASTER_CMD_BURST_RECEIVE_CONT);

        // If at last byte, go to final state
        if ( *i2c3_read_count <= 1 )
          *i2c3_int_state = STATE_READ_FINAL;

        break;
      }
    case STATE_READ_FINAL:
      {
        #ifdef DEBUG
        //UARTprintf("in STATE_READ_FINAL\n");
        #endif
        // Save last byte
        **i2c3_read_buffer = ROM_I2CMasterDataGet(I2C3_BASE);

        ROM_I2CMasterControl(I2C3_BASE, I2C_MASTER_CMD_BURST_RECEIVE_FINISH);
        *i2c3_int_state = STATE_IDLE;
        break;
      }
    }
}
