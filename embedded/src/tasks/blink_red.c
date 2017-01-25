/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "tasks/include/blink_red.h"


static void blink_red_task(void* params) {

  for (;;) {

    blink_rgb(RED_LED, 1);
    vTaskDelay(250 / portTICK_RATE_MS);
  }

}

bool blink_red_init(void) {
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


  if ( xTaskCreate(blink_red_task, (const portCHAR *)"Blink red", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;

  }
  return false;
}

