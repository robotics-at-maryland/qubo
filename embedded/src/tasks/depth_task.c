/*
 * Nathaniel Renegar
 * R@M 2019
 * naterenegar@gmail.com
 */

#include "tasks/include/depth_task.h"

bool depth_task_init() {
    if ( xTaskCreate(depth_task, (const portCHAR *)"Depth", 128, NULL,
                     tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
    return true;
  }
  return false;
}


static void depth_task(void *params) {
    float depth;

    ms5837_init(I2C1_BASE);
    /* ms5837_setModel(SENSOR_MODEL); */
    /* ms5837_setFluidDensity(I2C1_BASE, 1029); // 1029 kg/m^3 (Salt water) */

    for(;;) {

        // Try using if having speed issues. This only reads and calculates
        // first order pressure without temperature compensation

/*		ms5837_readPressureNoCalculate(I2C1_BASE);
        ms5837_simplePressureCalculate();
        depth = ms5837_depth(I2C1_BASE); */

        // This calculates depth from second order pressure with
        // temperature compensation

        /*
          ms5837_read(I2C_BUS);
          depth = ms5837_depth(I2C_BUS);
        */

    }
}
