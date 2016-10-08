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
 * Declaration of public functions that handle the board's
 * Nested Vectored Interrupt Controller (NVIC).
 *
 * @author Jernej Kovacic
 */


#ifndef _NVIC_H_
#define _NVIC_H_

#include <stdio.h>

/* MAximum value for priority, see page 124 of the Data Sheet: */
#define MAX_PRIORITY       ( 7 )


void nvic_enableInterrupt(uint8_t irq);

void nvic_disableInterrupt(uint8_t irq);

void nvic_setPriority(uint8_t irq, uint8_t pri);

uint8_t nvic_getPriority(uint8_t irq);

#endif  /* _NVIC_H_ */
