/* sgillen - this will be all the arduino code for qubo */

#include "debug.h"

#include <Wire.h>
#include "PCA9685.h"

#define NUM_THRUSTERS 8

#define THRUSTER_NEUTRAL 1285
/// THRUSTER_NEUTRAL - 256
#define THRUSTER_MIN 1029U
// THRUSTER_NETURAL + 256
#define THRUSTER_MAX 1541U

PCA9685 pca;


void setup() {
  Serial.begin(115200);

  //configure the PCA
  Wire.begin(); // Initiate the Wire library

  pca.init();

  for ( int i = 0; i < NUM_THRUSTERS; i++ ) {
    pca.thrusterSet(i, THRUSTER_NEUTRAL);
  }
  
  delay(2000);

  for ( int i = 0 ; i < NUM_THRUSTERS; i++ ) {
    pca.thrusterSet(i, 1350U);
  }
}




void loop() {

}

