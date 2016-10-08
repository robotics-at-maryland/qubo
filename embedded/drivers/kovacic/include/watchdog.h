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
 * Declaration of public functions that handle
 * both watchdog timers.
 *
 * @author Jernej Kovacic
 */


#ifndef _WATCHDOG_H_
#define _WATCHDOG_H_


#include <stdint.h>


/**
 * An enumeration with supported exception types
 * when a watchdog counter reaches 0.
 */
typedef enum _watchdog_exception
{
    WDEX_NMI,           /** Non-maskable interrupt (NMI) */
    WDEX_IRQ,           /** Standard interrupt */
    WDEX_NMI_RESET,     /** NMI on first time-out, reset on second */
    WDEX_IRQ_RESET      /** Standard int. on first time-out, reset on second */
} WatchdogException;


/**
 * Required prototype for watchdog triggered interrupt requests.
 */
typedef void (*WatchdogIntHandler_t)(void);


void wd_enableWd(uint8_t wd);

void wd_disableWd(uint8_t wd);

void  wd_start(uint8_t wd);

void wd_reset(uint8_t wd);

void wd_enableNvicIntr(void);

void wd_disableNvicIntr(void);

void wd_setIntPriority(uint8_t pri);

void wd_clearInterrupt(uint8_t wd);

uint32_t wd_getValue(uint8_t wd);

void wd_registerIntHandler(uint8_t wd, WatchdogIntHandler_t isr);

void wd_unregisterIntHandler(uint8_t wd);

void wd_reload(uint8_t wd);

void wd_config(uint8_t wd, uint32_t loadValue, WatchdogException ex);

#endif  /* _WATCHDOG_H_ */
