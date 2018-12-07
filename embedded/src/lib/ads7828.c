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
    /* buffer_out[2] = {0, 0}; */
    /* writeI2C(device, _i2caddr, buffer_out, 2); */
}
