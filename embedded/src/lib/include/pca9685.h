/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Ported from https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library


/***************************************************
  This is a library for our Adafruit 16-channel PWM & Servo driver

  Pick one up today in the adafruit shop!
  ------> http://www.adafruit.com/products/815

  These displays use I2C to communicate, 2 pins are required to
  interface. For Arduino UNOs, thats SCL -> Analog 5, SDA -> Analog 4

  Adafruit invests time and resources providing this open source code,
  please support Adafruit and open-source hardware by purchasing
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.
  BSD license, all text above must be included in any redistribution
 ****************************************************/

#ifndef _ADAFRUIT_PWMServoDriver_H
#define _ADAFRUIT_PWMServoDriver_H

#include "lib/include/query_i2c.h"
#include <math.h>

#define PCA9685_SUBADR1 0x2
#define PCA9685_SUBADR2 0x3
#define PCA9685_SUBADR3 0x4

#define PCA9685_MODE1 0x0
#define PCA9685_MODE2 0x1
#define PCA9685_PRESCALE 0xFE

#define LED0_ON_L 0x6
#define LED0_ON_H 0x7
#define LED0_OFF_L 0x8
#define LED0_OFF_H 0x9

#define ALLLED_ON_L 0xFA
#define ALLLED_ON_H 0xFB
#define ALLLED_OFF_L 0xFC
#define ALLLED_OFF_H 0xFD



void pca9685_begin(uint32_t device, uint8_t addr);
void pca9685_reset(uint32_t device);
void pca9685_setPWMFreq(uint32_t device, float freq);
void pca9685_setPWM(uint32_t device, uint8_t num, uint16_t on, uint16_t off);
void pca9685_setPin(uint32_t device, uint8_t num, uint16_t val, bool invert);


static uint8_t _i2caddr;

#endif
