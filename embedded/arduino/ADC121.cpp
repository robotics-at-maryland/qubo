#include "ADC121.h"
#include <Wire.h>

#define ADC121_ADDR 0x50

ADC121::ADC121() {}

int ADC121::getData() {
  unsigned int data[2];
  // Start I2C Transmission
  Wire.beginTransmission(ADC121_ADDR);
  // Calling conversion result register, 0x00(0)
  Wire.write(0x00);
  // Stop I2C transmission
  Wire.endTransmission();

  // Request 2 bytes of data
  Wire.requestFrom(ADC121_ADDR, 2);

  // Read 2 bytes of data
  // raw_adc msb, raw_adc lsb
  if(Wire.available() == 2)
    {
      data[0] = Wire.read();
      data[1] = Wire.read();
    }
  delay(300);
  // Convert the data to 12 bits
  return ((data[0] & 0x0F) * 256) + data[1];
}
