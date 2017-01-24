#include "include/example_blink.h"

bool example_blink_init(void) {
  ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
  while(!ROM_SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOF))
    {
    }

  //
  // Configure the GPIO port for the LED operation.
  //
  ROM_GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, RED_LED|BLUE_LED|GREEN_LED);

  if ( xTaskCreate(example_blink_task, (const portCHAR *)"Example", 200, NULL, tskIDLE_PRIORITY + 1, NULL)
      != pdTRUE) {

    return true;

  }
  return false;
}


static void example_blink_task(void* params) {

  for (;;) {
    ROM_GPIOPinWrite(GPIO_PORTF_BASE, RED_LED|GREEN_LED|BLUE_LED, RED_LED);
    ROM_SysCtlDelay(5000000);
    ROM_GPIOPinWrite(GPIO_PORTF_BASE, RED_LED|GREEN_LED|BLUE_LED, 0);
    ROM_SysCtlDelay(5000000);
  }

}
