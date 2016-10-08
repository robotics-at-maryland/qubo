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
 * Declaration of public functions that manipulate all 3 LEDs of
 * the Texas Instruments TM4C123GLX Launchpad directly,
 * without using general GPIO functions.
 *
 * @author Jernej Kovacic
 */

#ifndef _LED_H_
#define _LED_H_

#include <stdint.h>

/**
 * Bit masks for all three LEDs
 */
#define LED_RED       ( 0x00000002 )
#define LED_BLUE      ( 0x00000004 )
#define LED_GREEN     ( 0x00000008 )


void led_config(void);

void led_allOff(void);

void led_on(uint32_t leds);

void led_off(uint32_t leds);

#endif  /* _LED_H_ */
