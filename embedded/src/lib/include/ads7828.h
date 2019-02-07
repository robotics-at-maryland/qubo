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
static const uint8_t DIFFERENTIAL  = 0 << 7; // SD = 0
static const uint8_t SINGLE_ENDED = 1 << 7; // SD = 1

// PD Options
static const uint8_t REFERENCE_OFF = 0 << 3; // PD1 = 0
static const uint8_t REFERENCE_ON = 1 << 3; // PD1 = 1
static const uint8_t ADC_OFF = 0 << 2; // PD0 = 0
static const uint8_t ADC_ON = 1 << 2; // PD0 = 1

/*
 * ADDRESS BYTE: | 1 | 0 | 0 | 1 | 0 | A0 | A1 | R/W |
 * The 5 MSB's are factory hard-coded to 10010, 
 * A1 and A0 are set by two pins on the ADC and must be addressed
 * in the byte accordingly. 
 */ 

static const uint8_t DEFAULT_ADDRESS = 0x48
 
void ads7828_begin(uint32_t device, uint8_t addr);
uint16_t ads7828_readChannel(uint32_t device, uint8_t channel);

static uint8_t _i2caddr;

#endif
