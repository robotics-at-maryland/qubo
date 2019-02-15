/*
 * Jeremy Weed
 * jweed262@umd.edu
 * R@M 2018
 *
 * Nate Renegar
 * nrenegar@cs.umd.edu
 * R@M 2019  
 * 
 * Source: https://github.com/4-20ma/i2c_adc_ads7828/
 */

/*  
 * ADS7828 Header 
 * ADS7828 Datasheet: http://www.ti.com/lit/ds/symlink/ads7828.pdf
 * 
 */

#ifndef _ADS_7828_
#define _ADS_7828_

#include "lib/include/query_i2c.h"
#include <stdint.h>

/*
 * COMMAND BYTE: | SD | C2 | C1 | C0 | PD1 | PD0 | X | X |
 * SD: 0 -> Differential Inputs
 * SD: 1 -> Single-ended Inputs
 * C2 - C0: Channel Selection
 * PD1 - 0: Power-Down Selection
 * X: Unused
 * 
 * PD0: 0 -> A/D Converter OFF, 1 -> A/D Converter ON
 * PD1: 0 -> Internal Reference OFF, 1 -> Internal Reference ON
 */ 

// These will be OR'd together to form a COMMAND BYTE
// SD Options
#define DIFFERENTIAL 0x00 // SD = 0
#define SINGLE_ENDED 0x80 // SD = 1

// PD Options
#define REFERENCE_OFF 	0x00 // PD1 = 0
#define REFERENCE_ON  	0x08 // PD1 = 1
#define ADC_OFF 	0x00 // PD0 = 0
#define ADC_ON 		0x04 // PD0 = 1

/*
 * ADDRESS BYTE: | 1 | 0 | 0 | 1 | 0 | A1 | A0 | R/W |
 * The 5 MSB's are factory hard-coded to 10010, 
 * A1 and A0 are set by two pins on the ADC and must be addressed
 * in the byte accordingly. (On the power board, A1 and A0 are connected
 * to GND, corresponding to 0 0)
 * R/W is controlled by the I2C library
 */ 

#define DEFAULT_ADDRESS 0x48 // | A1 | A0 | = | 0 | 0 | 
 
void ads7828_begin(uint32_t device, uint8_t addr);
uint8_t ads7828_setChannel(uint8_t channel);
uint16_t ads7828_readChannel(uint32_t device, uint8_t channel);

static uint8_t _i2caddr = DEFAULT_ADDRESS;
static uint8_t _settings = (SINGLE_ENDED | REFERENCE_ON | ADC_ON); 
static uint8_t _command_byte;
static const uint8_t _se_channels[8] = {0x00, 0x40, 0x10, 0x50, 0x20, 0x60, 0x30, 0x70};

#endif
