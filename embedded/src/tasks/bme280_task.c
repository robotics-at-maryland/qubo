/*
 * Kyle Montemayor
 * kmonte@umd.edu
 * R@M 2017
 */
#include "tasks/include/bme280_task.h"
#include "lib/include/printfloat.h"

bool bme280_task_init() {
  if ( xTaskCreate(bme280_task_loop, (const portCHAR *)"BME280 Task", 256, NULL, tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

static void bme280_task_loop(void *params) {

  // garbage data
  uint8_t reg = 0xF0;
  uint8_t buffer = 0xAA;



  #ifdef DEBUG
  UARTprintf("Starting task\n");
  #endif

  if ( !bme280_begin(I2C0_BASE) ) {
    #ifdef DEBUG
    UARTprintf("bme280 amnessia\n");
    #endif
    //while(1){}
  }
  #ifdef DEBUG
  UARTprintf("initialized sensor\n");
  #endif

  char string[8];
  float a = 3;
  //ftoa(a, string, 5);
  //ROM_I2CMasterSlaveAddrSet(I2C0_BASE, 0x3C, false);




    a = bme280_readTemperature(I2C0_BASE);
    write_uart_wrapper(NULL, &a, sizeof(a));
    #ifdef DEBUG
    UARTprintf("finished function call, now in i2c_test task\n");
    #endif
    //ftoa(a, string, 5);
    //sprintf(string, "%f", a);

    #ifdef DEBUG
    //UARTprintf("%s\n", string);
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
  for (;;) {
  }
}
