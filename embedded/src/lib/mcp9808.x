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
bool begin(uint32_t device, uint8_t addr) {

  // default case
  if ( addr == 0 )
    _i2caddr = MCP9808_I2CADDR_DEFAULT;
  else
    _i2caddr = addr;

  uint8_t buffer[2];
  uint16_t result = 0;

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
float readTempC(uint32_t device)
{
  uint8_t buffer[2];
  uint16_t t = 0;

  readI2C(device, _i2caddr, MCP9808_REG_AMBIENT_TEMP, buffer, 2);
  ARR_TO_16(t, buffer);
  //uint16_t t = read16(MCP9808_REG_AMBIENT_TEMP);

  float temp = t & 0x0FFF;
  temp /=  16.0;
  if (t & 0x1000) temp -= 256;

  return temp;
}



//*************************************************************************
// Set Sensor to Shutdown-State or wake up (Conf_Register BIT8)
// 1= shutdown / 0= wake up
//*************************************************************************

int shutdown_wake(uint32_t device, uint8_t sw_ID )
{
  uint8_t buffer[2];
  uint16_t conf_shutdown ;
  uint16_t conf_register;
  readI2C(device, _i2caddr, MCP9808_REG_CONFIG, buffer, 2);
  ARR_TO_16(conf_register, buffer);
  //= read16(MCP9808_REG_CONFIG);
  if (sw_ID == 1)
  {
      conf_shutdown = conf_register | MCP9808_REG_CONFIG_SHUTDOWN ;
      writeI2C(device, _i2caddr, MCP9808_REG_CONFIG, (uint8_t *)&conf_shutdown, 2);
      //write16(MCP9808_REG_CONFIG, conf_shutdown);
  }
  if (sw_ID == 0)
  {
      conf_shutdown = conf_register ^ MCP9808_REG_CONFIG_SHUTDOWN ;
      writeI2C(device, _i2caddr, MCP9808_REG_CONFIG, (uint8_t *)&conf_shutdown, 2);
      //write16(MCP9808_REG_CONFIG, conf_shutdown);
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
