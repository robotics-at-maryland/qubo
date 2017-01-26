/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "tasks/include/blink_blue.h"


static void blink_blue_task(void* params) {

  for (;;) {

    blink_rgb(BLUE_LED, 1);
    vTaskDelay(1000 / portTICK_RATE_MS);
  }

}

bool blink_blue_init(void) {
  // Setup code done in main already 
  /*
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
  while(!ROM_SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOF))
    {
    }

  //
  // Configure the GPIO port for the LED operation.
  //
  ROM_GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, RED_LED|BLUE_LED|GREEN_LED);
  */


  if ( xTaskCreate(blink_blue_task, (const portCHAR *)"Blink blue", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;

  }
  return false;
}

