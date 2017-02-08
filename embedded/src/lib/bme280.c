/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// ported from https://github.com/adafruit/Adafruit_BME280_Library

#include "lib/include/bme280.h"

// Public
bool bme280_begin(uint32_t device) {
  // Make this an array so we can reuse it for the writes later
  uint8_t buffer[2];
  buffer[0] = BME280_REGISTER_CHIPID;

  #ifdef DEBUG
  UARTprintf("in bme280_begin\n");
  #endif

  queryI2C(device, BME280_ADDRESS, &(buffer[0]), 1, &(buffer[1]), 1);

  #ifdef DEBUG
  UARTprintf("After first query\n");
  #endif

  if ( buffer[1] != 0x60 )
    return false;

  readCoefficients(device);

  // 16x oversampling
  // Set before CONTROL_meas (DS 5.4.3)
  buffer[0] = BME280_REGISTER_CONTROLHUMID;
  buffer[1] = 0x05;
  writeI2C(device, BME280_ADDRESS, buffer, 2, false);

  // 16x oversampling, normal mode
  buffer[0] = BME280_REGISTER_CONTROL;
  buffer[1] = 0xB7;
  writeI2C(device, BME280_ADDRESS, buffer, 2, false);

  return true;
}

float bme280_readTemperature(uint32_t device) {
  int32_t var1, var2;
  uint8_t reg = BME280_REGISTER_TEMPDATA;

  uint8_t adc_T_ptr[3];

  // Zero this out so 3 bytes can properly be converted to a 32bit int
  int32_t adc_T = 0;

  // Query the BME280 with reg, and read back 3 bytes
  queryI2C(device, BME280_ADDRESS, &reg, 1, adc_T_ptr, 3);

  // Make the 3 bytes into a 32 bit int
  ARR_TO_32(adc_T, adc_T_ptr);

  adc_T >>= 4;

  var1  = ((((adc_T>>3) - ((int32_t)_bme280_calib.dig_T1 <<1))) *
           ((int32_t)_bme280_calib.dig_T2)) >> 11;

  var2  = (((((adc_T>>4) - ((int32_t)_bme280_calib.dig_T1)) *
             ((adc_T>>4) - ((int32_t)_bme280_calib.dig_T1))) >> 12) *
           ((int32_t)_bme280_calib.dig_T3)) >> 14;

  t_fine = var1 + var2;

  float T  = (t_fine * 5 + 128) >> 8;
  return T/100;
}

float bme280_readPressure(uint32_t device) {
  int64_t var1, var2, p;

  bme280_readTemperature(device); // must be done first to get t_fine

  uint8_t adc_P_ptr[3];
  int32_t adc_P = 0;

  uint8_t reg = BME280_REGISTER_PRESSUREDATA;
  queryI2C(device, BME280_ADDRESS, &reg, 1, adc_P_ptr, 3);

  // Make the 3 bytes into a 32 bit
  ARR_TO_32(adc_P, adc_P_ptr);

  adc_P >>= 4;

  var1 = ((int64_t)t_fine) - 128000;
  var2 = var1 * var1 * (int64_t)_bme280_calib.dig_P6;
  var2 = var2 + ((var1*(int64_t)_bme280_calib.dig_P5)<<17);
  var2 = var2 + (((int64_t)_bme280_calib.dig_P4)<<35);
  var1 = ((var1 * var1 * (int64_t)_bme280_calib.dig_P3)>>8) +
    ((var1 * (int64_t)_bme280_calib.dig_P2)<<12);
  var1 = (((((int64_t)1)<<47)+var1))*((int64_t)_bme280_calib.dig_P1)>>33;

  if (var1 == 0) {
    return 0;  // avoid exception caused by division by zero
  }
  p = 1048576 - adc_P;
  p = (((p<<31) - var2)*3125) / var1;
  var1 = (((int64_t)_bme280_calib.dig_P9) * (p>>13) * (p>>13)) >> 25;
  var2 = (((int64_t)_bme280_calib.dig_P8) * p) >> 19;

  p = ((p + var1 + var2) >> 8) + (((int64_t)_bme280_calib.dig_P7)<<4);
  return (float)p/256;
}

float bme280_readHumidity(uint32_t device) {

  bme280_readTemperature(device); // must be done first to get t_fine

  int32_t adc_H = 0;
  uint8_t adc_H_ptr[2];

  uint8_t reg = BME280_REGISTER_HUMIDDATA;
  queryI2C(device, BME280_ADDRESS, &reg, 1, adc_H_ptr, 2);

  // Make the 2 bytes to a 32bit
  ARR_TO_16(adc_H, adc_H_ptr);

  int32_t v_x1_u32r;

  v_x1_u32r = (t_fine - ((int32_t)76800));

  v_x1_u32r = (((((adc_H << 14) - (((int32_t)_bme280_calib.dig_H4) << 20) -
                  (((int32_t)_bme280_calib.dig_H5) * v_x1_u32r)) + ((int32_t)16384)) >> 15) *
               (((((((v_x1_u32r * ((int32_t)_bme280_calib.dig_H6)) >> 10) *
                    (((v_x1_u32r * ((int32_t)_bme280_calib.dig_H3)) >> 11) + ((int32_t)32768))) >> 10) +
                  ((int32_t)2097152)) * ((int32_t)_bme280_calib.dig_H2) + 8192) >> 14));

  v_x1_u32r = (v_x1_u32r - (((((v_x1_u32r >> 15) * (v_x1_u32r >> 15)) >> 7) *
                             ((int32_t)_bme280_calib.dig_H1)) >> 4));

  v_x1_u32r = (v_x1_u32r < 0) ? 0 : v_x1_u32r;
  v_x1_u32r = (v_x1_u32r > 419430400) ? 419430400 : v_x1_u32r;
  float h = (v_x1_u32r>>12);
  return  h / 1024.0;
}

/**************************************************************************/
/*!
  Calculates the altitude (in meters) from the specified atmospheric
  pressure (in hPa), and sea-level pressure (in hPa).
  @param  seaLevel      Sea-level pressure in hPa
  @param  atmospheric   Atmospheric pressure in hPa
*/
float bme280_readAltitude(uint32_t device, float seaLevel) {
  // Equation taken from BMP180 datasheet (page 16):
  //  http://www.adafruit.com/datasheets/BST-BMP180-DS000-09.pdf

  // Note that using the equation from wikipedia can give bad results
  // at high altitude.  See this thread for more information:
  //  http://forums.adafruit.com/viewtopic.php?f=22&t=58064

  float atmospheric = bme280_readPressure(device) / 100.0F;
  return 44330.0 * (1.0 - pow(atmospheric / seaLevel, 0.1903));
}

/**************************************************************************/
/*!
  Calculates the pressure at sea level (in hPa) from the specified altitude
  (in meters), and atmospheric pressure (in hPa).
  @param  altitude      Altitude in meters
  @param  atmospheric   Atmospheric pressure in hPa
*/
/**************************************************************************/
float bme280_seaLevelForAltitude(uint32_t device, float altitude, float atmospheric) {
  // Equation taken from BMP180 datasheet (page 17):
  //  http://www.adafruit.com/datasheets/BST-BMP180-DS000-09.pdf

  // Note that using the equation from wikipedia can give bad results
  // at high altitude.  See this thread for more information:
  //  http://forums.adafruit.com/viewtopic.php?f=22&t=58064

  return atmospheric / pow(1.0 - (altitude/44330.0), 5.255);
}

// Private
static void readCoefficients(uint32_t device) {
  uint8_t reg;
  uint8_t buffer[2];

  reg = BME280_REGISTER_DIG_T1;
  // Read 2 bytes save into calib
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // The LE function does this after reading
  ARR_TO_16(_bme280_calib.dig_T1, buffer);
  LE(_bme280_calib.dig_T1);

  reg = BME280_REGISTER_DIG_T2;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_T2, buffer);
  LE(_bme280_calib.dig_T2);

  reg = BME280_REGISTER_DIG_T3;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_T3, buffer);
  LE(_bme280_calib.dig_T3);

  reg = BME280_REGISTER_DIG_P1;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  ARR_TO_16(_bme280_calib.dig_P1, buffer);
  LE(_bme280_calib.dig_P2);

  reg = BME280_REGISTER_DIG_P2;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P2, buffer);
  LE(_bme280_calib.dig_P2);

  reg = BME280_REGISTER_DIG_P3;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P3, buffer);
  LE(_bme280_calib.dig_P3);

  reg = BME280_REGISTER_DIG_P4;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P4, buffer);
  LE(_bme280_calib.dig_P4);

  reg = BME280_REGISTER_DIG_P5;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P5, buffer);
  LE(_bme280_calib.dig_P5);

  reg = BME280_REGISTER_DIG_P6;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P6, buffer);
  LE(_bme280_calib.dig_P6);

  reg = BME280_REGISTER_DIG_P7;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P7, buffer);
  LE(_bme280_calib.dig_P7);

  reg = BME280_REGISTER_DIG_P8;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P8, buffer);
  LE(_bme280_calib.dig_P8);

  reg = BME280_REGISTER_DIG_P9;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_P9, buffer);
  LE(_bme280_calib.dig_P9);

  // Just reading one byte, so save directly to it
  reg = BME280_REGISTER_DIG_H1;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &(_bme280_calib.dig_H1), 1);

  reg = BME280_REGISTER_DIG_H2;
  queryI2C(device, BME280_ADDRESS, &reg, 1, buffer, 2);
  // Signed version
  ARR_TO_S16(_bme280_calib.dig_H2, buffer);
  LE(_bme280_calib.dig_H2);

  reg = BME280_REGISTER_DIG_H3;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &(_bme280_calib.dig_H3), 1);

  // They do some weird stuff here, just copying it
  uint8_t temp;
  reg = BME280_REGISTER_DIG_H4;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &(_bme280_calib.dig_H4), 1);
  reg = BME280_REGISTER_DIG_H4 + 1;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &temp, 1);
  temp = temp & 0xF;
  _bme280_calib.dig_H4 = (_bme280_calib.dig_H4 << 4 ) | temp;

  reg = BME280_REGISTER_DIG_H5;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &(_bme280_calib.dig_H5), 1);
  reg = BME280_REGISTER_DIG_H5 + 1;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &temp, 1);
  _bme280_calib.dig_H5 = ( temp << 4 ) | ( _bme280_calib.dig_H5 >> 4);

  reg = BME280_REGISTER_DIG_H6;
  queryI2C(device, BME280_ADDRESS, &reg, 1, &(_bme280_calib.dig_H6), 1);
  _bme280_calib.dig_H6 = (int8_t)_bme280_calib.dig_H6;
}
