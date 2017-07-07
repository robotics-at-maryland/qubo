/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Ported code from https://github.com/bluerobotics/BlueRobotics_MS5837_Library/

/* Blue Robotics Arduino MS5837-30BA Pressure/Temperature Sensor Library
------------------------------------------------------------
 
Title: Blue Robotics Arduino MS5837-30BA Pressure/Temperature Sensor Library
Description: This library provides utilities to communicate with and to
read data from the Measurement Specialties MS5837-30BA pressure/temperature 
sensor.
Authors: Rustom Jehangir, Blue Robotics Inc.
         Adam Šimko, Blue Robotics Inc.
-------------------------------
The MIT License (MIT)
Copyright (c) 2015 Blue Robotics Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-------------------------------*/

#ifndef MS5837_H_BLUEROBOTICS
#define MS5837_H_BLUEROBOTICS

#include <math.h>

#include <FreeRTOS.h>
#include <task.h>

#include "lib/include/query_i2c.h"

#define MS5837_ADDR               0x76
#define MS5837_RESET              0x1E
#define MS5837_ADC_READ           0x00
#define MS5837_PROM_READ          0xA0
#define MS5837_CONVERT_D1_8192    0x4A
#define MS5837_CONVERT_D2_8192    0x5A

static const float Pa = 100.0f;
static const float bar = 0.001f;
static const float mbar = 1.0f;

static const uint8_t MS5837_30BA = 0;
static const uint8_t MS5837_02BA = 1;

// Public
void ms5837_init(uint32_t device);

/** Set model of MS5837 sensor. Valid options are MS5837::MS5837_30BA (default)
  * and MS5837::MS5837_02BA.
  */
void ms5837_setModel(uint8_t model);

/** Provide the density of the working fluid in kg/m^3. Default is for
  * seawater. Should be 997 for freshwater.
  */
void ms5837_setFluidDensity(uint32_t device, float density);

/** The read from I2C takes up for 40 ms, so use sparingly is possible.
  */
void ms5837_read(uint32_t device);

/** This function loads the datasheet test case values to verify that
  *  calculations are working correctly. No example checksum is provided
  *  so the checksum test may fail.
  */
void ms5837_readTestCase(uint32_t device);

/** Pressure returned in mbar or mbar*conversion rate.
  */
float ms5837_pressure(uint32_t device, float conversion);

/** Temperature returned in deg C.
  */
float ms5837_temperature(uint32_t device);

/** Depth returned in meters (valid for operation in incompressible
  *  liquids only. Uses density that is set for fresh or seawater.
  */
float ms5837_depth(uint32_t device);

/** Altitude returned in meters (valid for operation in air only).
  */
float ms5837_altitude(uint32_t device);


// Private
static uint16_t C[8];
static uint32_t D1, D2;
static int32_t TEMP;
static int32_t P;
static uint8_t _model;

static float fluidDensity = 1029;

	/** Performs calculations per the sensor data sheet for conversion and
	 *  second order compensation.
	 */
static void calculate(void);

static uint8_t crc4(uint16_t *n_prom);

#endif
