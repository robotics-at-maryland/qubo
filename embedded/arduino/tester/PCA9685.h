#ifndef _PCA9685_H_
#define _PCA9685_H_

#include "debug.h"

#include "Arduino.h"

#define BUFFER_SIZE 128

class PCA9685 {
 public:

  PCA9685();

  void init();

  void thrustersOff();
  void thrusterSet(uint8_t thruster, uint16_t command);

 private:
  int freq;
  uint8_t read8(uint8_t addr);
  void write8(uint8_t addr, uint8_t d);
};

#endif
