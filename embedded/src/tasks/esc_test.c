#include "tasks/include/esc_task.h"
#include "lib/include/printfloat.h"
#include <stdio.h>

bool esc_test_init() {
  if ( xTaskCreate(esc_test_task, (const portCHAR *)"I2C Test", 256, NULL, tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

static void esc_test_task(void *params) {

  // garbage data
  uint8_t reg = 0xF0;
  uint8_t buffer = 0xAA;


  #ifdef DEBUG
  UARTprintf("Starting ESC task\n");
  #endif

  char string[8];
  uint32_t test = 10;
  float a = 10.0;
  //ftoa(a, string, 5);
  //ROM_I2CMasterSlaveAddrSet(I2C0_BASE, 0x3C, false);


  for (;;) {
      **i2c0_write_buffer = 0;
      // Put the current data in the buffer
      ROM_I2CMasterDataPut(I2C0_BASE, **i2c0_write_buffer);
      // Decrement the count pointer
      *i2c0_write_count = *i2c0_write_count - 1;
      // Point to the next byte
      *i2c0_write_buffer = *i2c0_write_buffer + 1;
      // Send the data
      ROM_I2CMasterControl(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);
      
      
  }
}

