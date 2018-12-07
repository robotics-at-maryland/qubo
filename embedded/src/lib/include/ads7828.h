/*
 * Jeremy Weed
 * jweed262@umd.edu
 * R@M 2018
 *
 * Source: https://github.com/4-20ma/i2c_adc_ads7828/
 */

#ifndef _ADS_7828_
#define _ADS_7828_

#include "lib/include/query_i2c.h"
#include <stdint.h>


void ads7828_begin(uint32_t device, uint8_t addr);
uint16_t ads7828_readChannel(uint32_t device, uint8_t channel);

static uint8_t _i2caddr;

#endif
