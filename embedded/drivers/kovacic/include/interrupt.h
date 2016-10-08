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
 * Declarations of functions that enable or disable activation of
 * exceptions/interrupts and define the minimum priority for
 * exception processing.
 *
 * @author Jernej Kovacic
 */

#ifndef _INTERRUPT_H_
#define _INTERRUPT_H_

#include <stdint.h>


void intr_enableInterrupts(void);

void intr_disableInterrupts(void);

void intr_enableException(void);

void intr_disableException(void);

void intr_setBasePriority(uint8_t pri);


#endif   /* _INTERRUPT_H_ */
