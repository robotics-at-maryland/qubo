#include "tasks/include/i2c_test.h"

bool i2c_test_init() {
  if ( xTaskCreate(i2c_test_task, (const portCHAR *)"I2C Test", 256, NULL, tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

static void i2c_test_task(void *params) {

  // garbage data
  uint8_t reg = 0xF0;
  uint8_t buffer = 0xAA;

  #ifdef DEBUG
  UARTprintf("bme280_begin\n");
  #endif

  if ( !bme280_begin(I2C3_BASE) ) {
    #ifdef DEBUG
    UARTprintf("error in bme280 begin\n");
    #endif
  }

  float a = -1;

  for (;;) {
		#ifdef DEBUG
    UARTprintf("sending data on i2c");
		#endif
    a = bme280_readTemperature(I2C3_BASE);
    #ifdef DEBUG
    UARTprintf("Temp: %f\n");
    #endif
  }
}

