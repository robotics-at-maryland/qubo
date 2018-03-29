#include "tasks/include/i2c_test.h"
#include "lib/include/printfloat.h"
#include <stdio.h>

#define PCA_ADDRESS 0x70

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

  //if ( !mcp9808_begin(I2C0_BASE, 0) ) {
    #ifdef DEBUG
    //UARTprintf("error in mcp9808 begin\n");
    #endif
  //}

  pca9685_begin(I2C0_BASE, PCA_ADDRESS);
  pca9685_setPWMFreq(I2C0_BASE, 1600); //originally 1600
  for (int i = 0; i < 8; i++)
    pca9685_setPWM(I2C0_BASE, i, 0, 0);
   
  #ifdef DEBUG
  UARTprintf("initialized sensors\n");
  #endif

  char string[8];
  uint32_t test = 10;
  float a = 10.0;
  //ftoa(a, string, 5);
  //ROM_I2CMasterSlaveAddrSet(I2C0_BASE, 0x3c, false);

  vTaskDelay(500);

  for (int i = 0; i < 8; i++)
    pca9685_setPWM(I2C0_BASE, i, 0, 1520);

  for (;;) {
    
    //for (int i = 0; i < 7; i++) 
      //pca9685_setPWM(I2C0_BASE, i, 0, 1520);

    #ifdef DEBUG
    //UARTprintf("-----WAKE UP-----\n");
    #endif
    //mcp9808_shutdown_wake(I2C0_BASE, 0);

    vTaskDelay(500);

    #ifdef DEBUG
    //UARTprintf("----READ TEMP-----\n");
    #endif
    //a = mcp9808_readTempC(I2C0_BASE);
    //sprintf(string, "%+6.*f", 3, a);
    #ifdef DEBUG
    //UARTprintf("/\/\/\ TEMP DATA: %x\n", PFLOAT(a));
    //UARTprintf("-----SHUT DOWN-----\n");
    #endif

    //mcp9808_shutdown_wake(I2C0_BASE, 1);
    //ftoa(a, string, 5);
    //sprintf(string, "%f", a);

    #ifdef DEBUG
    //UARTprintf("%s\n", string);
    #endif

    //vTaskDelay(3250);

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

