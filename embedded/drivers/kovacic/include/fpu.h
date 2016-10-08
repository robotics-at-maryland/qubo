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
 * the Floating-Point Unit (FPU).
 *
 * @author Jernej Kovacic
 */


#ifndef _FPU_H_
#define _FPU_H_

#include <stdbool.h>


/**
 * An enumeration with supported FPU Half Precision Modes.
 */
typedef enum _FpuHalfPrecisionMode
{
    FPUHPM_IEEE,               /** IEEE representation */
    FPUHPM_ALTERNATIVE         /** Cortex-M alternative representation */
} FpuHalfPrecisionMode;


/**
 * An enumeration with supported NaN modes.
 */
typedef enum _FpuNanMode
{
    FPU_NAN_PROPAGATE,        /** IEEE representation */
    FPU_NAN_DEFAULT           /** Cortex-M alternative representation */
} FpuNanMode;


/**
 * An enumeration with supported rounding modes.
 */
typedef enum _FpuRMode
{
    FPU_RMODE_RN,            /** Round to Nearest mode */
    FPU_RMODE_RP,            /** Round towards Plus Infinity mode */
    FPU_RMODE_RM,            /** Round towards Minus Infinity mode */
    FPU_RMODE_RZ             /** Round towards Zero mode */
} FpuRMode;



void fpu_enable(void);

void fpu_disable(void);

void fpu_enableStacking(void);

void fpu_enableLazyStacking(void);

void fpu_disableStacking(void);

void fpu_setHalfPrecisionMode(FpuHalfPrecisionMode mode);

void fpu_setNanMode(FpuNanMode mode);

void fpu_setFlushToZero(bool fz);

void fpu_setRoundingMode(FpuRMode mode);

#endif  /* _FPU_H_ */
