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

#include "lib/include/pca9685.h"

void pca9685_begin(uint32_t device, uint8_t addr) {
  _i2caddr = addr;
  //pca9685_reset(device);
}


void pca9685_reset(uint32_t device) {
  //uint8_t buffer1[1] = {PCA9685_MODE1};
  //writeI2C(device, _i2caddr, buffer1, 1);
  //uint8_t buffer2[1] = {0x01};
  //writeI2C(device, _i2caddr, buffer2, 1);
}

void pca9685_setPWMFreq(uint32_t device, float freq) {
  freq *= 0.9;  // Correct for overshoot in the frequency setting (see issue #11).
  float prescaleval = 25000000;
  prescaleval /= 4096;
  prescaleval /= freq;
  prescaleval -= 1;
  #ifdef DEBUG
    //Serial.print("Estimated pre-scale: "); Serial.println(prescaleval);
  #endif
  #ifdef DEBUG
    //Serial.print("Final pre-scale: "); Serial.println(prescale);
  #endif

  uint8_t mode[2] = {PCA9685_MODE1, 0x21};
  uint8_t prescale[2] = {PCA9685_PRESCALE, floor(prescaleval + 0.5)};

  mode[1] |= 0x10;
  writeI2C(device, _i2caddr, mode, 2);
  writeI2C(device, _i2caddr, prescale, 2);
  mode [1] &= ~0x10;
  writeI2C(device, _i2caddr, mode, 2);
}

void pca9685_setPWM(uint32_t device, uint8_t num, uint16_t on, uint16_t off) {

  #ifdef DEBUG
  UARTprintf("pca9685: Setting PWM %d -> %d\n", on, off);
  #endif

  uint8_t buffer[5] = {LED0_ON_L+4*num, on, on>>8, off, off>>8};
  writeI2C(device, _i2caddr, buffer, 5);
}

// Sets pin without having to deal with on/off tick placement and properly handles
// a zero value as completely off.  Optional invert parameter supports inverting
// the pulse for sinking to ground.  Val should be a value from 0 to 4095 inclusive.
void pca9685_setPin(uint32_t device, uint8_t num, uint16_t val, bool invert)
{
  // Clamp value between 0 and 4095 inclusive.
  // min(val, 4095)
  if ( val > 4095 )
    val = 4095;
  if (invert) {
    if (val == 0) {
      // Special value for signal fully on.
      pca9685_setPWM(device, num, 4096, 0);
    }
    else if (val == 4095) {
      // Special value for signal fully off.
      pca9685_setPWM(device, num, 0, 4096);
    }
    else {
      pca9685_setPWM(device, num, 0, 4095-val);
    }
  }
  else {
    if (val == 4095) {
      // Special value for signal fully on.
      pca9685_setPWM(device, num, 4096, 0);
    }
    else if (val == 0) {
      // Special value for signal fully off.
      pca9685_setPWM(device, num, 0, 4096);
    }
    else {
      pca9685_setPWM(device, num, 0, val);
    }
  }
}
