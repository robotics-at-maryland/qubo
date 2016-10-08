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
 * Declarations of public functions that
 * handle the built-in System Timer (SysTick).
 *
 * @author Jernej Kovacic
 */

#ifndef _SYSTICK_H_
#define _SYSTICK_H_

#include <stdint.h>
#include <stdbool.h>


void systick_disable(void);

void systick_enable(void);

void systick_setSource(bool systemClock);

bool systick_countSet(void);

void systick_setReload(uint32_t value);

void systick_clear(void);

uint32_t systick_getCurrentValue(void);

void systick_enableInterrupt(void);

void systick_disableInterrupt(void);

void systick_clearInterrupt(void);

void systick_setPriority(uint8_t pri);

void systick_config(uint32_t reload);

#endif  /* _SYSTICK_H_ */
