/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "tasks/include/example_blink.h"


static void example_blink_task(void* params) {

  for (;;) {

    blink_rgb(GREEN_LED, 1);
  }

}

bool example_blink_init(void) {
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


  if ( xTaskCreate(example_blink_task, (const portCHAR *)"Example", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;

  }
  return false;
}

