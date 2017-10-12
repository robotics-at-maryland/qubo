/* sgillen - this will be all the arduino code for qubo */

#define DEBUG

#include <Wire.h>
#include "MS5837.h"

#define PCA9685_MODE1 0x0
#define PCA9685_PRESCALE 0xFE

#define PCA9Address 0xFF // Device address in which is also included the 8th bit for selecting the mode, read in this case.
#define allCall 0xE0 //address for "all call" which should turn all LEDs on

#define LED0_ON_L 0x6
#define LED0_ON_H 0x7
#define LED0_OFF_L 0x8
#define LED0_OFF_H 0x9

#define BUFFER_SIZE 512 //may need to change
#define NUM_THRUSTERS 8

// Time(ms) arduino waits without hearing from jetson before turning off thrusters
#define ALIVE_TIMEOUT 9999999

// Character sent to the jetson on connect and reconnect
#define CONNECTED "C"

// __________________________________________________________________________________________

char buffer[BUFFER_SIZE]; //this is the buffer where we store incoming text from the computer
uint16_t serialBufferPos;
int freq = 1600; //pretty arbitrary, max is 1600
unsigned long alive; // keeps the current time
boolean timedout = false; // if the arduino has timed out
MS5837 sensor;
//Adafruit_PWMServoDriver pwm;

// __________________________________________________________________________________________


//used by the PCA
uint8_t read8(uint8_t addr) {
  Wire.beginTransmission(PCA9Address);
  Wire.write(addr);
  Wire.endTransmission();

  Wire.requestFrom((uint8_t)PCA9Address, (uint8_t)1);
  return Wire.read();
}

//used by the PCA
void write8(uint8_t addr, uint8_t d) {
  Wire.beginTransmission(PCA9Address);
  Wire.write(addr);
  Wire.write(d);
  Wire.endTransmission();
}


void thrustersOff() {
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

void setup() {
  Serial.begin(115200);
  serialBufferPos = 0;

  //configure the PCA
  Wire.begin(); // Initiate the Wire library


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

  // Depth sensor
  sensor.init();

  sensor.setFluidDensity(997); // kg/m^3 (997 freshwater, 1029 for seawater)

  // Done setup, so send connected command
  //Serial.println(CONNECTED);
  alive = millis();
  delay(1);
}

//processes and sends thrusters commands to the PCA
void thrusterCmd() {

  char* thrusterCommands[NUM_THRUSTERS];

  for (int i = 0; i < NUM_THRUSTERS - 1; i++) {

    thrusterCommands[i] = strtok(NULL, ","); //remember buffer is global, strok still remembers that we are reading from it
    #ifdef DEBUG
    Serial.println(thrusterCommands[i]);
    #endif

  }
  thrusterCommands[NUM_THRUSTERS] = strtok(NULL, "!"); //last token is the ! not the ,

  uint16_t check = atoi(thrusterCommands[0]);

  for (int i = 0; i < NUM_THRUSTERS; i++) {

    uint16_t off = atoi(thrusterCommands[i]);

    Wire.beginTransmission(PCA9Address);
    Wire.write(LED0_ON_L + 4 * i);
    Wire.write(0);
    Wire.write(0);
    Wire.write(off);
    Wire.write(off >> 8);
    Wire.endTransmission();

  }

  // Send back the first command
  Serial.println(check);

}

// Placeholder, needs to get depth from I2C, then println it to serial
void getDepth() {
  sensor.read();
  float depth = sensor.depth();
  Serial.println(depth);
}

void loop() {

  //this is all the stuff we do while the jetson is talking to us
  if (Serial.available() > 0) {

    // If just reconnected from a timeout, tell jetson its connected again
    if ( timedout ) {
      //Serial.println(CONNECTED);
      timedout = false;
    }

    // Read next byte from serial into buffer
    buffer[serialBufferPos] = Serial.read();

    #ifdef DEBUG
     Serial.print("buffer is: ");
     Serial.println(buffer);
    #endif

    // Check if we've reached exclamation
    if (buffer[serialBufferPos] == '!') {

      //prot tells us what to do
      char* prot = strtok(buffer, ",");

      //check if something about the packet is malformed enough that strok fails
      if ( !prot[0] ) {
        Serial.println("B1");
      }
      // Handle specific commands
      else if ( prot[0] == 't' ) {
        #ifdef DEBUG
        //Serial.println("Thrusters on");
        #endif
          thrusterCmd();
        }
      else if ( prot[0] == 'd' ) {
        #ifdef DEBUG
        //Serial.println("Get depth");
        #endif
          getDepth();
      }
      else {
          Serial.println("B2");
          Serial.println(prot[0]);
      }

      // Reset buffer position
      serialBufferPos = 0;
      buffer[0] = 0;

    }

    else {
      #ifdef DEBUG

      Serial.print("Buffer pos ");
      Serial.println(serialBufferPos);
      Serial.println(buffer[serialBufferPos]);

      #endif
      serialBufferPos++;
    }

    // update the timer
    alive = millis();

  }//end Serial.available is

  // Timeout checking
  else {

    unsigned long current_time = millis();
    // If the time wrapped around, can't just subtract them, take the difference from max and then the current_time
    if ( current_time <= alive ){
      unsigned long max_long = (unsigned long) -1;
      if ( ((max_long - alive) + current_time ) >= ALIVE_TIMEOUT ){
        #ifdef DEBUG
        Serial.println(max_long - alive + current_time);
        Serial.println(ALIVE_TIMEOUT);
        #endif
        if (!timedout) {
          #ifdef DEBUG
          Serial.println("Overflow Timed out, thrusters off");
          #endif
          thrustersOff();
          timedout = true;
        }
      }
    }
    // If time hasn't wrapped around, just take their difference
    else if (( current_time - alive) >= ALIVE_TIMEOUT ) {
      if (!timedout) {
        #ifdef DEBUG
        Serial.println("Timed out, thrusters off");
        #endif
        thrustersOff();
        timedout = true;
      }
    }
  }
  delay(1);
  //here we put code that we need to run with or without the jetsons attached
}

