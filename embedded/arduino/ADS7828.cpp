/*
 Library for the TI ADS7828 12bits i2C ADC 
 Datasheet: http://www.ti.com/lit/ds/symlink/ads7828.pdf

 Deskwizard (03/16/2013)

 ------------------------------------------------------------------------------
 Library information -  Current Library version: 0.1d - March 16th 2013
 Tested working on IDE version 1.0.3

 This is a basic library to use the ADS7828 ADC.
 
 Both single ended and differential modes are currently available.
 Please read the information below if you plan to use differential mode.
 
 Both Internal and external voltage reference mode are available.
 Please read the datasheet if you are to use the internal reference!

 ------------------------------------------------------------------------------    
 Chip Wiring
 All connections necessary unless specifically mentionned.
 
 Connect GND pin to Ground
 Connect +VDD pin to 5v
 Connect SCL pin to Arduino SCL (Analog 5)
 Connect SDA pin to Arduino SDA (Analog 4)
 Connect A0 pin to either ground or 5v (set device address accordingly)
 Connect A1 pin to either ground or 5v (set device address accordingly) 
 Connect COM pin to ground (Single-ended mode common channel)
 
 CH0 - CH7 are obviously the ADC input channels
 
 ------------------------------------------------------------------------------    
 Function list
 
  init(INT)*                  Initialize the ADC and the I2C bus with internal voltage reference. (* Use one or the other)
  init()*                     Initialize the ADC and the I2C bus with external voltage reference. (* Use one or the other)
  read(channel, SD|DF)        Read the specified ADC channel (0-7) in either single ended (SD) or differential (DF) mode.
                              Reading a channel will return a value between 0-4095
                              
 ------------------------------------------------------------------------------    
 Using differential mode
 
 When using differential mode, it is important to pay attention to the channel naming.
 The channel configuration is contained in Table II on page 11 of the datasheet.
 
 It is important to respect the polarity of the channels when reading in differential mode
 or your readings will be 0.

 For example:

  Differential Channel 0:      Positive side on Channel 0, Negative side on Channel 1
  Differential Channel 1:      Positive side on Channel 1, Negative side on Channel 0
  Differential Channel 2:      Positive side on Channel 2, Negative side on Channel 3
  Differential Channel 3:      Positive side on Channel 3, Negative side on Channel 2
  etc...
*/

#include "Arduino.h"
#include "ADS7828.h"
#include "Wire.h"

int ads_address;				// ADC I2C address
bool ads_vref_int_enabled = 0;  // default voltage reference is external

// command address for the channels, allows 0-7 channel mapping in the correct order
unsigned char channels[8] = {0x00, 0x40, 0x10, 0x50, 0x20, 0x60, 0x30, 0x70};

ADS7828::ADS7828(unsigned char _address){
  ads_address = _address;   // Set ADC i2c address to the one passed to the function
}

void ADS7828::init(){
  Wire.begin();   // Initialize I2C Bus
}

void ADS7828::init(boolean _vref){
  Wire.begin();   // Initialize I2C Bus
  ads_vref_int_enabled = _vref;  // Set vref trigger to specified value (Internal or external)
}

unsigned int ADS7828::read(unsigned char channel, bool mode)
{
  unsigned char command = 0;		// Initialize command variable
  unsigned int reading = 0;			// Initialize reading variable

  command = channels[channel];      // put required channel address in command variable

  if (mode){
    command = command ^ 0x80; 		// Enable Single-ended mode (toggle MSB, SD bit in datasheet)
  }
  if (ads_vref_int_enabled){
    command = command ^ 0x08; 	    // Enable internal voltage reference if ads_vref_int_enabled = 1
  }

  Wire.beginTransmission(ads_address); 	// Send a start or repeated start command with a slave address and the R/W bit set to '0' for writing.
  Wire.write(command);      			// Then send a command byte for the register to be read.
  Wire.endTransmission();				// Send stop command

  Wire.requestFrom(ads_address, 2);		// Request 2 bytes from the ADC

  if(2 <= Wire.available())    		// if two bytes were received
  {
    reading = Wire.read();     		// receive high byte
    reading = reading << 8;    		// shift high byte to be high 8 bits
    reading |= Wire.read();    		// receive low byte into lower 8 bits
  }
  return reading;					// return the full 12 bits reading from the ADC channel
}
