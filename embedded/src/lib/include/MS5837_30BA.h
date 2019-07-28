/*
 * Nate Renegar
 * naterenegar@umd.edu
 * R@M 2019  
 * 
 */

/*  
 * MS5837-30BA Header 
 * MS5837-30BA Product page: https://bluerobotics.com/store/sensors-sonars-cameras/sensors/bar30-sensor-r1/ 
 */

#ifndef _MS5837_30BA_H
#define _MS5837_30BA_H

#include "lib/include/query_i2c.h"
#include <stdint.h>


void MS5837_30BA_begin(uint32_t device, uint8_t addr); 
void MS5837_30BA_read_prom();



#endif

