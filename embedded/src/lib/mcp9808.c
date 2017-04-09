/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#include "lib/include/mcp9808.h"

/**************************************************************************/
/*!
    @brief  Setups the HW
*/
/**************************************************************************/
bool mcp9808_begin(uint32_t device, uint8_t addr) {

  #ifdef DEBUG
  UARTprintf("In mcp9808 begin\n");
  #endif
  // default case
  if ( addr == 0 )
    _i2caddr = MCP9808_I2CADDR_DEFAULT;
  else
    _i2caddr = addr;

  uint8_t buffer[2];
  uint16_t result = 0;

  #ifdef DEBUG
  UARTprintf("Before readi2c\n");
  #endif

  readI2C(device, _i2caddr, MCP9808_REG_MANUF_ID, buffer, 2);
  ARR_TO_16(result, buffer);

  if ( result == 0x0054 )
    return false;

  readI2C(device, _i2caddr, MCP9808_REG_DEVICE_ID, buffer, 2);
  ARR_TO_16(result, buffer);

  if ( result == 0x0400 )
    return false;

  return true;
}

/**************************************************************************/
/*! 
    @brief  Reads the 16-bit temperature register and returns the Centigrade
            temperature as a float.

*/
/**************************************************************************/
float mcp9808_readTempC(uint32_t device) {
  uint8_t buffer[2];
  uint16_t t = 0;

  #ifdef DEBUG
  UARTprintf("In read temp\n");
  #endif
  readI2C(device, _i2caddr, MCP9808_REG_AMBIENT_TEMP, buffer, 2);
  ARR_TO_16(t, buffer);
#ifdef DEBUG
  UARTprintf("mcp9808: buffer[0] %x buffer[1] %x\nmcp9808: t %x\n", buffer[0], buffer[1], t);
#endif
  //uint16_t t = read16(MCP9808_REG_AMBIENT_TEMP);

  float temp = t & 0x0FFF;
  temp /=  16.0;
  if (t & 0x1000) temp -= 256;

  #ifdef DEBUG
  UARTprintf("about to return\n");
  #endif
  return temp;
}



//*************************************************************************
// Set Sensor to Shutdown-State or wake up (Conf_Register BIT8)
// 1= shutdown / 0= wake up
//*************************************************************************

int mcp9808_shutdown_wake(uint32_t device, uint8_t sw_ID ) {
  uint8_t buffer[3];
  uint16_t conf_shutdown ;
  uint16_t conf_register;
  buffer[0] = MCP9808_REG_CONFIG;
  // Fill last two parts of buffer
  readI2C(device, _i2caddr, MCP9808_REG_CONFIG, &(buffer[1]), 2);
  ARR_TO_16(conf_register, buffer);
  //= read16(MCP9808_REG_CONFIG);
  if (sw_ID == 1)
  {
#ifdef DEBUG
    UARTprintf("mcp9808: shutdown before sending 3 bytes\n");
#endif
      conf_shutdown = conf_register | MCP9808_REG_CONFIG_SHUTDOWN ;
      buffer[2] = (conf_shutdown >> 8);
      buffer[1] = (conf_shutdown & 0xFF);
      writeI2C(device, _i2caddr, buffer, 3);
#ifdef DEBUG
      UARTprintf("mcp9808: shutdown sent 3 bytes\n");
      UARTprintf("mcp9808: %x %x %x\n", buffer[0], buffer[1], buffer[2]);
#endif

      //write16(MCP9808_REG_CONFIG, conf_shutdown);
  }
  if (sw_ID == 0)
  {
    #ifdef DEBUG
    UARTprintf("mcp9808: wakeup before sending 3 bytes\n");
    #endif
      buffer[2] = (conf_shutdown >> 8);
      buffer[1] = (conf_shutdown & 0xFF);
      conf_shutdown = conf_register ^ MCP9808_REG_CONFIG_SHUTDOWN ;
      writeI2C(device, _i2caddr, buffer, 3);
      //write16(MCP9808_REG_CONFIG, conf_shutdown);
      #ifdef DEBUG
      UARTprintf("mcp9808: wakeup sent 3 bytes\n");
      UARTprintf("mcp9808: %x %x %x\n", buffer[0], buffer[1], buffer[2]);
      #endif
  }


  return 0;
}




/**************************************************************************/
/*! 
    @brief  Low level 16 bit read and write procedures!
*/
/**************************************************************************/

/*
void Adafruit_MCP9808::write16(uint8_t reg, uint16_t value) {
    Wire.beginTransmission(_i2caddr);
    Wire.write((uint8_t)reg);
    Wire.write(value >> 8);
    Wire.write(value & 0xFF);
    Wire.endTransmission();
}

uint16_t Adafruit_MCP9808::read16(uint8_t reg) {
  uint16_t val;

  Wire.beginTransmission(_i2caddr);
  Wire.write((uint8_t)reg);
  Wire.endTransmission();
  
  Wire.requestFrom((uint8_t)_i2caddr, (uint8_t)2);
  val = Wire.read();
  val <<= 8;
  val |= Wire.read();  
  return val;  
}
*/
