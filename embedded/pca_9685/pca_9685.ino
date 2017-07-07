/* 
 Code to talk to our PCA9685 PWM multiplexer using our arduino via I2c


//sgillen - stole a lot of this code from this adafruit repo https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library for a board they sell with this chip on it. 
//had to change the mode flags to get it to work using the internal oscilator

*/

#include <Wire.h>


#define PCA9685_MODE1 0x0
#define PCA9685_PRESCALE 0xFE

#define PCA9Address 0xFF // Device address in which is also included the 8th bit for selecting the mode, read in this case.
#define allCall 0xE0 //address for "all call" which should turn all LEDs on


#define LED0_ON_L 0x6
#define LED0_ON_H 0x7
#define LED0_OFF_L 0x8
#define LED0_OFF_H 0x9


int freq = 1400; //pretty arbitrary, max is 1600


uint8_t read8(uint8_t addr) {
  Wire.beginTransmission(PCA9Address);
  Wire.write(addr);
  Wire.endTransmission();

  Wire.requestFrom((uint8_t)PCA9Address, (uint8_t)1);
  return Wire.read();
}

void write8(uint8_t addr, uint8_t d) {
  Wire.beginTransmission(PCA9Address);
  Wire.write(addr);
  Wire.write(d);
  Wire.endTransmission();
}


void setup() {
  Wire.begin(); // Initiate th1e Wire library
  Serial.begin(9600);
  delay(100);

  
  
  Serial.print("Attempting to set freq ");
  Serial.println(freq);
  freq *= 0.9;  // Correct for overshoot in the frequency setting (see issue #11).
  float prescaleval = 25000000;
  prescaleval /= 4096;
  prescaleval /= freq;
  prescaleval -= 1;

  uint8_t prescale = floor(prescaleval + 0.5);
  
  uint8_t oldmode = read8(PCA9685_MODE1);
  uint8_t oldmode2 = read8(PCA9685_MODE1 + 1);

  Serial.println("old mode was");
  Serial.println(oldmode);
  Serial.println(oldmode2);
  
  uint8_t newmode = 0x21;
  write8(PCA9685_MODE1, newmode); // go to sleep
  write8(PCA9685_PRESCALE, prescale); // set the prescaler
  //write8(PCA9685_MODE1, oldmode);
  //delay(5);
  //write8(PCA9685_MODE1, oldmode | 0xa1); 


  oldmode = read8(PCA9685_MODE1);
  oldmode2 = read8(PCA9685_MODE1 + 1);

  Serial.println("new mode is");
  Serial.println(oldmode);
  Serial.println(oldmode2);

  
 
  
}
void loop() {
 

  uint16_t on = 1;
  uint16_t off = 2000;

  Wire.beginTransmission(PCA9Address);
  Wire.write(LED0_ON_L+4);
  Wire.write(on);
  Wire.write(on>>8);
  Wire.write(off);
  Wire.write(off>>8);
  Wire.endTransmission();

  delay(1000);

  Serial.println(read8(LED0_ON_L +4));
  Serial.println(read8(LED0_ON_H+4));

  Serial.println();

  Serial.println(read8(LED0_OFF_L+4));
  Serial.println(read8(LED0_OFF_H+4)); 
  
}
