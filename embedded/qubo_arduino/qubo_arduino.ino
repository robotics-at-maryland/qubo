/* sgillen - this will be all the arduino code for qubo */

#include "debug.h"

#include <Wire.h>
#include "MS5837.h"
#include "PCA9685.h"
#include "ADC121.h"

#define BUFFER_SIZE 512 //may need to change
#define NUM_THRUSTERS 8

// Time(ms) arduino waits without hearing from jetson before turning off thrusters
#define ALIVE_TIMEOUT 9999999

// Character sent to the jetson on connect and reconnect
#define CONNECTED "C"

// __________________________________________________________________________________________

#define LM35_PIN 0

char buffer[BUFFER_SIZE]; //this is the buffer where we store incoming text from the computer
uint16_t serialBufferPos;
unsigned long alive; // keeps the current time
boolean timedout = false; // if the arduino has timed out
PCA9685 pca;
MS5837 sensor;
ADC121 adc121;

void setup() {
  Serial.begin(115200);
  serialBufferPos = 0;

  //configure the PCA
  Wire.begin(); // Initiate the Wire library

  pca.init();

  pca.thrustersOff();

  // Depth sensor
  sensor.init();

  sensor.setFluidDensity(997); // kg/m^3 (997 freshwater, 1029 for seawater)

  // for lm35 https://playground.arduino.cc/Main/LM35HigherResolution
  analogReference(INTERNAL);

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

    pca.thrusterSet(i, off);
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

void getCurrent() {
  int data = adc121.getData();
  Serial.println(data);
}

void getTemp() {
  int val = analogRead(LM35_PIN);
  float temp = val / 9.31;
  Serial.println(temp);
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
      else if ( prot[0] == 'c' ) {
        #ifdef DEBUG
        Serial.println("Get temp");
        #endif
        getTemp();
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
          pca.thrustersOff();
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
        pca.thrustersOff();
        timedout = true;
      }
    }
  }
  delay(1);
  //here we put code that we need to run with or without the jetsons attached
}

