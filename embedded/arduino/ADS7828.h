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

#ifndef ADS7828_H
#define ADS7828_H

#include "Arduino.h"
#include <Wire.h>

#define EXT 0 // External VREF
#define INT 1 // Internal VREF

#define SD 1 // Single ended mode
#define DF 0 // Differential mode


class ADS7828
{


public:
	ADS7828(unsigned char _address);
	void init();
	void init(boolean _vref);
	unsigned int read(unsigned char channel, bool mode);

};


#endif
