/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "tasks/include/servo_test.h"


static void servo_test_task(void* params) {

  pca9685_begin(I2C0_BASE, 0x40);
  pca9685_setPWMFreq(I2C0_BASE, 1600);

  for (;;) {
    for (uint16_t i=0; i<4096; i += 8) {
      pca9685_setPWM(I2C0_BASE, 0, 0, (i % 4096 ));
      vTaskDelay(250);
    }
  }

}

bool servo_test_init(void) {

  if ( xTaskCreate(servo_test_task, (const portCHAR *)"Example", 128, NULL,
                   tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;

  }
  return false;
}

