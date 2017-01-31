/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _I2C0_MUTEX_H_
#define _I2C0_MUTEX_H_

#include <FreeRTOS.h>
#include <semphr.h>

extern SemaphoreHandle_t i2c0_mutex;

#endif
