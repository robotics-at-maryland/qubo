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
 * the board's System Controller (SysCtrl).
 *
 * @author Jernej Kovacic
 */


#ifndef _SYSCTL_H_
#define _SYSCTL_H_

#include <stdint.h>

int8_t sysctl_mcuRevision(void);

uint8_t sysctl_configSysClock(uint8_t div);

void sysctl_enableGpioPort(uint8_t port);

void sysctl_disableGpioPort(uint8_t port);

void sysctl_enableUart(uint8_t uartNr);

void sysctl_disableUart(uint8_t uartNr);

void sysctl_enableWatchdog(uint8_t wd);

void sysctl_disableWatchdog(uint8_t wd);

void sysctl_resetWatchdog(uint8_t wd);

#endif  /* _SYSCTL_H_ */
