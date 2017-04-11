#include "tasks/include/i2c_test.h"
#include "lib/include/printfloat.h"
#include "lib/include/write_uart.h"
#include <stdio.h>

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

  if ( !mcp9808_begin(I2C0_BASE, 0) ) {
    #ifdef DEBUG
    UARTprintf("error in mcp9808 begin\n");
    #endif
  }
  #ifdef DEBUG
  UARTprintf("initialized sensor\n");
  #endif

  char string[8];
  uint32_t test = 10;
  float a = 10.0;
  //ftoa(a, string, 5);
  //ROM_I2CMasterSlaveAddrSet(I2C0_BASE, 0x3C, false);


  for (;;) {


    #ifdef DEBUG
    UARTprintf("-----WAKE UP-----\n");
    #endif
    mcp9808_shutdown_wake(I2C0_BASE, 0);

    vTaskDelay(500);

    #ifdef DEBUG
    UARTprintf("----READ TEMP-----\n");
    #endif
    a = mcp9808_readTempC(I2C0_BASE);
    //sprintf(string, "%+6.*f", 3, a);
    #ifdef DEBUG
    UARTprintf("/\/\/\ TEMP DATA: %x\n", PFLOAT(a));
    UARTprintf("-----SHUT DOWN-----\n");
    #endif

    mcp9808_shutdown_wake(I2C0_BASE, 1);
    //ftoa(a, string, 5);
    //sprintf(string, "%f", a);

    #ifdef DEBUG
    //UARTprintf("%s\n", string);
    #endif

    vTaskDelay(3250);

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

