#include "tasks/include/i2c_test.h"
#include "lib/include/printfloat.h"
//#include <stdio.h>

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
  UARTprintf("Starting task\n");
  #endif

  if ( !mcp9808_begin(I2C0_BASE) ) {
    #ifdef DEBUG
    UARTprintf("error in mcp9808 begin\n");
    #endif
  }
  #ifdef DEBUG
  UARTprintf("initialized sensor\n");
  #endif

  char string[8];
  float a = 3;
  ftoa(a, string, 5);
  //ROM_I2CMasterSlaveAddrSet(I2C0_BASE, 0x3C, false);


  for (;;) {

    a = mcp9808_readTempC(I2C0_BASE);
    #ifdef DEBUG
    UARTprintf("finished function call, now in i2c_test task\n");
    #endif
    ftoa(a, string, 5);
    //sprintf(string, "%f", a);

    #ifdef DEBUG
    UARTprintf("%s\n", string);
    #endif
    /*
		#ifdef DEBUG
    UARTprintf("sending data on i2c");
		#endif
    a = bme280_readTemperature(I2C0_BASE);
    #ifdef DEBUG
    UARTprintf("Temp: %f\n");
    #endif
    */

  }
}

