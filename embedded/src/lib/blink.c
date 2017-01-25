/* Blink the RGB with given color and some number of times */

#include "lib/include/blink.h"

void blink_led(uint8_t color, uint8_t n) {

  if ( xSemaphoreTake(blink_mutex, 0) == pdFALSE ) {
    return;
  }

  uint8_t i = 0;
  TickType_t ticktime = xTaskGetTickCount();

  while ( i < n ) {
    ROM_GPIOPinWrite(GPIO_PORTF_BASE, RED_LED|GREEN_LED|BLUE_LED, color);
    //ROM_SysCtlDelay(5000000);
    vTaskDelayUntil(&ticktime, BLINK_RATE / portTICK_RATE_MS);

    ROM_GPIOPinWrite(GPIO_PORTF_BASE, RED_LED|GREEN_LED|BLUE_LED, 0);
    //ROM_SysCtlDelay(5000000);
    vTaskDelayUntil(&ticktime, BLINK_RATE / portTICK_RATE_MS);
    i++;
  }

  xSemaphoreGive(blink_mutex);
}
