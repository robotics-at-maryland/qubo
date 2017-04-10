/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// Ported from arduino

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

// Set to true to print some debug messages, or false to disable them.
#define ENABLE_DEBUG_OUTPUT false

void pca9685_begin(uint32_t device, uint8_t addr) {
  _i2caddr = addr;
  pca9685_reset(device);
}


void pca9685_reset(uint32_t device) {
  uint8_t buffer[2] = {PCA9685_MODE1, 0x0};
  writeI2C(device, _i2caddr, buffer, 2);
  // write8(PCA9685_MODE1, 0x0);
}

void pca9685_setPWMFreq(uint32_t device, float freq) {
  //Serial.print("Attempting to set freq ");
  //Serial.println(freq);
  freq *= 0.9;  // Correct for overshoot in the frequency setting (see issue #11).
  float prescaleval = 25000000;
  prescaleval /= 4096;
  prescaleval /= freq;
  prescaleval -= 1;
  if (ENABLE_DEBUG_OUTPUT) {
    //Serial.print("Estimated pre-scale: "); Serial.println(prescaleval);
  }
  //uint8_t prescale = floor(prescaleval + 0.5);
  if (ENABLE_DEBUG_OUTPUT) {
    //Serial.print("Final pre-scale: "); Serial.println(prescale);
  }

  // 0x00 are temp values that will be reassigned. buffer[1] = newmode, buffer[5] = oldmode
  uint8_t buffer[6] = {PCA9685_MODE1, 0x00,
                       PCA9685_PRESCALE, floor(prescaleval + 0.5),
                       PCA9685_MODE1, 0x00};

  //uint8_t oldmode; // = read8(PCA9685_MODE1);
  readI2C(device, _i2caddr, PCA9685_MODE1, &(buffer[5]), 1);
  //uint8_t newmode = (oldmode&0x7F) | 0x10; // sleep
  buffer[1] = (buffer[5] & 0x7F) | 0x10; // sleep
  // Unoptimized
  //uint8_t buffer[6] = {PCA9685_MODE1, newmode, PCA9685_PRESCALE, prescale, PCA9685_MODE1, oldmode}
  writeI2C(device, _i2caddr, buffer, 6);
  //  write8(PCA9685_MODE1, newmode); // go to sleep
  //  write8(PCA9685_PRESCALE, prescale); // set the prescaler
  //  write8(PCA9685_MODE1, oldmode);
  //delay(5);
  vTaskDelay(5);
  buffer[1] = buffer[5] | 0xa1;
  writeI2C(device, _i2caddr, buffer, 2);
  //write8(PCA9685_MODE1, oldmode | 0xa1);  //  This sets the MODE1 register to turn on auto increment.
                                          // This is why the beginTransmission below was not working.
  //  Serial.print("Mode now 0x"); Serial.println(read8(PCA9685_MODE1), HEX);
}

void pca9685_setPWM(uint32_t device, uint8_t num, uint16_t on, uint16_t off) {
  //Serial.print("Setting PWM "); Serial.print(num); Serial.print(": "); Serial.print(on); Serial.print("->"); Serial.println(off);

  uint8_t buffer[5] = {LED0_ON_L+4*num, on, on>>8, off, off>>8};
  writeI2C(device, _i2caddr, buffer, 5);
  /*
  WIRE.beginTransmission(_i2caddr);
  WIRE.write(LED0_ON_L+4*num);
  WIRE.write(on);
  WIRE.write(on>>8);
  WIRE.write(off);
  WIRE.write(off>>8);
  WIRE.endTransmission();
  */
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

/*
uint8_t Adafruit_PWMServoDriver::read8(uint8_t addr) {
  WIRE.beginTransmission(_i2caddr);
  WIRE.write(addr);
  WIRE.endTransmission();

  WIRE.requestFrom((uint8_t)_i2caddr, (uint8_t)1);
  return WIRE.read();
}

void Adafruit_PWMServoDriver::write8(uint8_t addr, uint8_t d) {
  WIRE.beginTransmission(_i2caddr);
  WIRE.write(addr);
  WIRE.write(d);
  WIRE.endTransmission();
}
*/
