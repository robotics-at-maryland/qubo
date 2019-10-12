#include "tasks/include/thruster_task.h"
#include <stdio.h>

bool thruster_task_init() {
  if ( xTaskCreate(thruster_task, (const portCHAR *)"Thruster", 256, NULL,
                   tskIDLE_PRIORITY + 3, NULL) != pdTRUE) {
    return true;
  }
  return false;
}

static void thruster_task(void *params) {
  struct Thruster_Set thruster_set;
  bool init = false;

  pca9685_begin(I2C3_BASE, PCA_ADDR);
  pca9685_setPWMFreq(I2C3_BASE, PWM_FREQ);
  blink_rgb(BLUE_LED | GREEN_LED, 1);

  for (;;) {
    // wait indefinitely for something to come over the buffer
    /* xMessageBufferReceive(thruster_message_buffer, (void*)&thruster_set, */
    /*                       sizeof(thruster_set), portMAX_DELAY); */
    xQueueReceive(thruster_message_buffer, (void*)&thruster_set,
                  //portMAX_DELAY);
					pdMS_TO_TICKS(10));

    pca9685_setPWM(I2C3_BASE, thruster_set.thruster_id,  0, THRUSTER_SCALE(thruster_set.throttle));
	
  }
}
