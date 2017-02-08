/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// ported from https://github.com/adafruit/Adafruit_BME280_Library

// TODO:
// Not sure if macros should be big or little endian


#ifndef _BME280_H_
#define _BME280_H_

#include "lib/include/query_i2c.h"

#include <math.h>

/*=========================================================================
  I2C ADDRESS/BITS
  -----------------------------------------------------------------------*/
#define BME280_ADDRESS 0x77
/*=========================================================================*/

/*=========================================================================
  REGISTERS
  -----------------------------------------------------------------------*/

// Macro that mimics original _LE functions
#define LE(X) ( X = (X >> 8) | (X << 8) )

// Macro to convert a 2-array of uint8_t to signed int16_t
#define ARR_TO_S16(X,Y) ( X = (int16_t)(Y[0] | (Y[1] << 8)) )

// Macro to convert a 2-array of uint8_t to uint16_t
#define ARR_TO_16(X,Y) ( X = Y[0] | (Y[1] << 8) )

// Macro to convert a 3-array of uint8_t to uint32_t
#define ARR_TO_32(X,Y) ( X = Y[0] | (Y[1] << 8) | (Y[2] << 16) )

#define BME280_REGISTER_DIG_T1 				0x88
#define BME280_REGISTER_DIG_T2 				0x8A
#define BME280_REGISTER_DIG_T3 				0x8C

#define BME280_REGISTER_DIG_P1 				0x8E
#define BME280_REGISTER_DIG_P2 				0x90
#define BME280_REGISTER_DIG_P3 				0x92
#define BME280_REGISTER_DIG_P4 				0x94
#define BME280_REGISTER_DIG_P5 				0x96
#define BME280_REGISTER_DIG_P6 				0x98
#define BME280_REGISTER_DIG_P7 				0x9A
#define BME280_REGISTER_DIG_P8 				0x9C
#define BME280_REGISTER_DIG_P9 				0x9E

#define BME280_REGISTER_DIG_H1 				0xA1
#define BME280_REGISTER_DIG_H2 				0xE1
#define BME280_REGISTER_DIG_H3 				0xE3
#define BME280_REGISTER_DIG_H4 				0xE4
#define BME280_REGISTER_DIG_H5 				0xE5
#define BME280_REGISTER_DIG_H6 				0xE7

#define BME280_REGISTER_CHIPID 				0xD0
#define BME280_REGISTER_VERSION 			0xD1
#define BME280_REGISTER_SOFTRESET 		0xE0

#define BME280_REGISTER_CAL26 				0xE1  // R calibration stored in 0xE1-0xF0

#define BME280_REGISTER_CONTROLHUMID	0xF2
#define BME280_REGISTER_CONTROL 			0xF4
#define BME280_REGISTER_CONFIG 				0xF5
#define BME280_REGISTER_PRESSUREDATA 	0xF7
#define BME280_REGISTER_TEMPDATA 			0xFA
#define BME280_REGISTER_HUMIDDATA 		0xFD

/*=========================================================================
  CALIBRATION DATA
  -----------------------------------------------------------------------*/

typedef struct
{
  uint16_t dig_T1;
  int16_t  dig_T2;
  int16_t  dig_T3;

  uint16_t dig_P1;
  int16_t  dig_P2;
  int16_t  dig_P3;
  int16_t  dig_P4;
  int16_t  dig_P5;
  int16_t  dig_P6;
  int16_t  dig_P7;
  int16_t  dig_P8;
  int16_t  dig_P9;

  uint8_t  dig_H1;
  int16_t  dig_H2;
  uint8_t  dig_H3;
  int16_t  dig_H4;
  int16_t  dig_H5;
  int8_t   dig_H6;
} bme280_calib_data;

// Public

bool bme280_begin(uint32_t device);

float bme280_readTemperature(uint32_t device);
float bme280_readPressure(uint32_t device);
float bme280_readHumidity(uint32_t device);
float bme280_readAltitude(uint32_t device, float seaLevel);
float bme280_seaLevelForAltitude(uint32_t device, float altitude, float atmospheric);

// Private
static void readCoefficients(uint32_t device);

static bme280_calib_data _bme280_calib;

static int32_t t_fine;

#endif
