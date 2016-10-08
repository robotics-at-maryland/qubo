/*
Copyright 2014, Jernej Kovacic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file
 *
 * Declaration of public functions that manipulate both
 * switches of the Texas Instruments TM4C123GLX Launchpad
 * directly, without using general GPIO functions.
 *
 * @author Jernej Kovacic
 */


#ifndef _SWITCH_H_
#define _SWITCH_H_

#include <stdint.h>

#include "gpio.h"

/**
 * Bit masks for both switches
 */

#define SWITCH1       ( 0x00000010 )
#define SWITCH2       ( 0x00000001 )


void switch_config(void);

uint8_t switch_statusSw1(void);

uint8_t switch_statusSw2(void);

uint8_t switch_statusBoth(void);

void switch_enableSwInt(uint8_t sw);

void switch_disableSwInt(uint8_t sw);

void switch_registerIntrHandler(uint8_t sw, GpioPortIntHandler_t isr);

void switch_unregisterIntrHandler(uint8_t sw);

void switch_clearIntr(uint8_t sw);

#endif  /* _SWITCH_H_ */
