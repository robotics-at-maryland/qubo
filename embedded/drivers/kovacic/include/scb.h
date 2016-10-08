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
 * System Control Block (SCB).
 *
 * @author Jernej Kovacic
 */


#ifndef _SCB_H_
#define _SCB_H_

#include <stdint.h>

void scb_setSysTickPriority(uint8_t pri);

void scb_setPendSvPriority(uint8_t pri);

void scb_triggerPendSv(void);

void scb_clearPendSv(void);

void scb_reset(void);

void scb_pendSysTickIntr(void);

void scb_unpendSysTickIntr(void);

uint8_t scb_activeException(void);

#endif  /* _SCB_H_ */
