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
 * Declaration of data structures and functions that
 * handle switches and LEDs.
 *
 * @author Jernej Kovacic
 */

#ifndef _LIGHTSHOW_H_
#define _LIGHTSHOW_H_

#include <stdint.h>

#include <task.h>


/**
 * A struct with parameters to be passed to the
 * task that handles the switch 1.
 */
typedef struct _Switch1TaskParam_t
{
    TaskHandle_t ledHandle;    /** Handle of the light show task */
} Switch1TaskParam_t;


/**
 * A struct with parameters to be passed to the
 * light show task that periodically controls LEDs.
 */
typedef struct _LightShowParam_t
{
    uint32_t delayMs;          /** Period of LED switching in milliseconds */
} LightShowParam_t;


int16_t lightshowInit(void);

void lightshowTask(void* params);

void sw1DsrTask(void* params);

#endif  /* _LIGHTSHOW_H_ */
