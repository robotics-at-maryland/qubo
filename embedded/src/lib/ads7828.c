/*
 * Jeremy Weed
 * jweed262@umd.edu
 * R@M 2018
 */

#include "lib/include/ads7828.h"

void ads7828_begin(uint32_t device, uint8_t addr) {
    _i2caddr = addr;
}

uint16_t ads7828_readChannel(uint32_t device, uint8_t channel) {	
	uint8_t data_in[2] = {0, 0};
	uint16_t converted_data_in = 0x0000;
	_command_byte = _settings | _se_channels[channel];

	writeI2C(device, _i2caddr, &(_command_byte), 1);
	readI2C(device, _i2caddr, 1, data_in, 2);

	converted_data_in = data_in[0];
	converted_data_in = (converted_data_in << 8) | data_in[1];

	return converted_data_in;	
}

