/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

#ifndef _MCP9808_H_
#define _MCP9808_H_

#include "lib/include/query_i2c.h"

// Macro to convert a 2-array of uint8_t to signed int16_t
//#define ARR_TO_S16(X,Y) ( X = (int16_t)(Y[0] | (Y[1] << 8)) )
#define ARR_TO_S16(X,Y) ( X = (int16_t)(Y[1] | (Y[0] << 8)) )

// Macro to convert a 2-array of uint8_t to uint16_t
//#define ARR_TO_16(X,Y) ( X = Y[0] | (Y[1] << 8) )
#define ARR_TO_16(X,Y) ( X = Y[1] | (Y[0] << 8) )

// Macro to convert a 3-array of uint8_t to uint32_t
//#define ARR_TO_32(X,Y) ( X = Y[0] | (Y[1] << 8) | (Y[2] << 16) )
#define ARR_TO_32(X,Y) ( X = Y[2] | (Y[1] << 8) | (Y[0] << 16) )

#define MCP9808_I2CADDR_DEFAULT        0x18
#define MCP9808_REG_CONFIG             0x01

#define MCP9808_REG_CONFIG_SHUTDOWN    0x0100
#define MCP9808_REG_CONFIG_CRITLOCKED  0x0080
#define MCP9808_REG_CONFIG_WINLOCKED   0x0040
#define MCP9808_REG_CONFIG_INTCLR      0x0020
#define MCP9808_REG_CONFIG_ALERTSTAT   0x0010
#define MCP9808_REG_CONFIG_ALERTCTRL   0x0008
#define MCP9808_REG_CONFIG_ALERTSEL    0x0004
#define MCP9808_REG_CONFIG_ALERTPOL    0x0002
#define MCP9808_REG_CONFIG_ALERTMODE   0x0001

#define MCP9808_REG_UPPER_TEMP         0x02
#define MCP9808_REG_LOWER_TEMP         0x03
#define MCP9808_REG_CRIT_TEMP          0x04
#define MCP9808_REG_AMBIENT_TEMP       0x05
#define MCP9808_REG_MANUF_ID           0x06
#define MCP9808_REG_DEVICE_ID          0x07

bool mcp9808_begin(uint32_t device, uint8_t addr);
float mcp9808_readTempF(uint32_t device);
float mcp9808_readTempC(uint32_t device);
int mcp9808_shutdown_wake(uint32_t device, uint8_t sw_ID);

static uint8_t _i2caddr;

#endif
