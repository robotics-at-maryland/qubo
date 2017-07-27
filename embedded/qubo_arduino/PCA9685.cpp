#include "PCA9685.h"
#include <Wire.h>

#define THRUSTER_NEUTRAL 1285
/// THRUSTER_NEUTRAL - 256
#define THRUSTER_MIN 1029U
// THRUSTER_NETURAL + 256
#define THRUSTER_MAX 1541U

#define PCA9685_MODE1 0x0
#define PCA9685_PRESCALE 0xFE

#define PCA9Address 0xFF // Device address in which is also included the 8th bit for selecting the mode, read in this case.
#define allCall 0xE0 //address for "all call" which should turn all LEDs on

#define LED0_ON_L 0x6
#define LED0_ON_H 0x7
#define LED0_OFF_L 0x8
#define LED0_OFF_H 0x9

#define NUM_THRUSTERS 8

PCA9685::PCA9685() {}

void PCA9685::init() {

  freq = 1600;

#ifdef DEBUG
  Serial.print("Attempting to set freq ");
  Serial.println(freq);
#endif
  freq *= 0.9;  // Correct for overshoot in the frequency setting (see issue #11).
  float prescaleval = 25000000;
  prescaleval /= 4096;
  prescaleval /= freq;
  prescaleval -= 1;

  uint8_t prescale = floor(prescaleval + 0.5);

  uint8_t oldmode = read8(PCA9685_MODE1);
  uint8_t oldmode2 = read8(PCA9685_MODE1 + 1);

#ifdef DEBUG
  Serial.println("old mode was");
  Serial.println(oldmode);
  Serial.println(oldmode2);
#endif

  uint8_t newmode = 0x21;
  write8(PCA9685_MODE1, newmode); // go to sleep
  write8(PCA9685_PRESCALE, prescale); // set the prescaler
  //write8(PCA9685_MODE1, oldmode);
  //delay(5);
  //write8(PCA9685_MODE1, oldmode | 0xa1);

  oldmode = read8(PCA9685_MODE1);
  oldmode2 = read8(PCA9685_MODE1 + 1);

#ifdef DEBUG
  Serial.println("new mode is");
  Serial.println(oldmode);
  Serial.println(oldmode2);
  Serial.println("PCA initialized");
#endif

  thrustersOff();
}

uint8_t PCA9685::read8(uint8_t addr) {
  Wire.beginTransmission(PCA9Address);
  Wire.write(addr);
  Wire.endTransmission();

  Wire.requestFrom((uint8_t)PCA9Address, (uint8_t)1);
  return Wire.read();
}

void PCA9685::write8(uint8_t addr, uint8_t d) {
  Wire.beginTransmission(PCA9Address);
  Wire.write(addr);
  Wire.write(d);
  Wire.endTransmission();
}

void PCA9685::thrustersOff() {
  // turn off thrusters
  for (int i = 0; i < NUM_THRUSTERS; i++) {
    Wire.beginTransmission(PCA9Address);
    Wire.write(LED0_ON_L + 4 * i);
    Wire.write(0);
    Wire.write(0);
    Wire.write(0);
    Wire.write(0);
    Wire.endTransmission();
  }
}

void PCA9685::thrusterSet(uint8_t thruster, uint16_t command) {

  Wire.beginTransmission(PCA9Address);
  Wire.write(LED0_ON_L + 4 * thruster);
  Wire.write(0);
  Wire.write(0);
  Wire.write(command);
  Wire.write(command >> 8);
  Wire.endTransmission();
}
