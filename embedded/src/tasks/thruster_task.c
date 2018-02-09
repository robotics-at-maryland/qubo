#include "tasks/include/thruster_task.h"
#include "lib/include/printfloat.h"
#include <stdio.h>

bool esc_test_init() {
  if ( xTaskCreate(esc_test_task, (const portCHAR *)"Thruster", 256, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

static void esc_test_task(void *params) {

#ifdef DEBUG
  UARTprintf("Starting ESC task\n");
#endif

  pca9685_begin(I2C_BUS, PCA_ADDR);
  pca9685_setPWM(I2C_BUS, PWM_FREQ);



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
