#include "tasks/include/thruster_task.h"
#include "lib/include/printfloat.h"
#include <stdio.h>

bool thruster_task_init() {
  if ( xTaskCreate(thruster_task, (const portCHAR *)"Thruster", 256, NULL,
                   tskIDLE_PRIORITY + 2, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

static void thruster_task(void *params) {

  #ifdef DEBUG
  UARTprintf("Starting ESC task\n");
  #endif

  /* blink_rgb(BLUE_LED | RED_LED,  1); */

  pca9685_begin(THRUSTER_I2C_BUS, PCA_ADDR);
  pca9685_setPWMFreq(THRUSTER_I2C_BUS, PWM_FREQ);

  for (uint8_t i = 0; i < 8; i++) {
    pca9685_setPWM(THRUSTER_I2C_BUS, i,  10, THRUSTER_SCALE(0) - 15);
  }

  /* blink_rgb(BLUE_LED, 1); */
  struct Thruster_Set thruster_set;
  for (;;) {
    // wait indefinitely for something to come over the buffer
    /* xMessageBufferReceive(thruster_message_buffer, (void*)&thruster_set, */
    /*                       sizeof(thruster_set), portMAX_DELAY); */
    xQueueReceive(thruster_message_buffer, (void*)&thruster_set,
                  portMAX_DELAY);

    for (uint8_t i = 0; i < 8; i++) {
      pca9685_setPWM(THRUSTER_I2C_BUS, i,  10, THRUSTER_SCALE(thruster_set.throttle[i]) - 15);
    }


  }
}
