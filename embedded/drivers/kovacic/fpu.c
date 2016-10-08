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
 * Implementation of the board's Floating-Point Unit
 * (FPU) functionality that uses the internal coprocessor
 * to perform arithmetics on single precision floating
 * point numbers.
 *
 * For more info about the FPU, see pp. 130 - 134 and
 * 194 - 199 of:
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stdbool.h>

#include "bsp.h"
#include "regutil.h"
#include "fpu.h"


/*
 * 32-bit registers of the Floating-Point Unit, relative of the
 * first FPU related register of the Peripheral Controller.
 * 
 * See page 137 of the Data Sheet.
 */
typedef struct _TM4C123G_FPU_REGS
{
    uint32_t FPU_CPAC;                    /* Coprocessor Access Control */
    const uint32_t Reserved1[106];        /* reserved */
    uint32_t FPU_FPCC;                    /* Floating-Point Context Control */
    uint32_t FPU_FPCA;                    /* Floating-Point Context Address */
    uint32_t FPU_FPDSC;                   /* Floating-Point Default Status Control */
} TM4C123G_FPU_REGS;


static volatile TM4C123G_FPU_REGS* const pReg =
   (volatile TM4C123G_FPU_REGS* const) ( BSP_FPU_BASE_ADDRESS );


/*
 * Convenience bit masks to enable/disable both built-in
 * coprocessors (CP10 and CP11). See page 195 of the Data Sheet.
 */
#define CPAC_CP11_MASK            ( 0x00C00000 )
#define CPAC_CP10_MASK            ( 0x00300000 )

/*
 * Flags that enable/disable stacking of coprocessor's
 * registers on interrupt events.
 * See pp. 196 - 197 of the Data Sheet.
 */
#define FPCC_LSPEN                ( 0x40000000 )
#define FPCC_ASPEN                ( 0x80000000 )


/*
 * Flags and bit masks to enable several modes.
 * See page 199 of the Data Sheet.
 */
#define FPDSC_AHP                 ( 0x04000000 )
#define FPDSC_DN                  ( 0x02000000 )
#define FPDSC_FZ                  ( 0x01000000 )
#define FPDSC_RMODE_MASK          ( 0x00C00000 )
#define FPDSC_RMODE_RN            ( 0x00000000 )
#define FPDSC_RMODE_RP            ( 0x00400000 )
#define FPDSC_RMODE_RM            ( 0x00800000 )
#define FPDSC_RMODE_RZ            ( 0x00C00000 )


/**
 * Enables the floating-point unit, allowing the
 * floating-point instructions to be executed.
 * This function must be called prior to performing
 * any hardware floating-point operations
 */
void fpu_enable(void)
{
    /*
     * FPU is enabled by setting of CP10 and CP11
     * bits in the CPAC register.
     * For more details, see page 195 of the Data Sheet.
     */

    HWREG_SET_BITS( pReg->FPU_CPAC, (CPAC_CP11_MASK | CPAC_CP10_MASK) );
}


/**
 * Disables the floating-point unit.
 */
void fpu_disable(void)
{
    /*
     * FPU is disabled by clearing of CP10 and CP11
     * bits in the CPAC register.
     * For more details, see page 195 of the Data Sheet.
     */

    HWREG_CLEAR_BITS( pReg->FPU_CPAC, (CPAC_CP11_MASK | CPAC_CP10_MASK) );
}


/**
 * Enables the stacking of floating-point registers when an
 * interrupt is handled. Space is reserved on the stack for
 * the floating-point context and the floating-point state is
 * saved into this stack space.  Upon return from the interrupt,
 * the floating-point context is restored.
 */
void fpu_enableStacking(void)
{
    /*
     * LSPEN flag of the FPCC register is cleared,
     * ASPEN flag is set.
     * For more details, see page 196 of the Data Sheet.
     */

    HWREG_CLEAR_BITS( pReg->FPU_FPCC, FPCC_LSPEN );
    HWREG_SET_BITS( pReg->FPU_FPCC, FPCC_ASPEN );
}


/**
 * Enables the lazy stacking of floating-point registers
 * when an interrupt is handled. When enabled, space is
 * reserved on the stack for the floating-point context,
 * but the floating-point state is not saved.  If a FP
 * instruction is executed from within the interrupt context,
 * the floating-point context is first saved into the space
 * reserved on the stack. On completion of the interrupt
 * handler, the floating-point context is only restored if
 * it was saved (as the result of executing a FP instruction).
 */
void fpu_enableLazyStacking(void)
{
    /*
     * LSPEN and ASPEN flags of the FPCC register are set.
     * For more details, see page 196 of the Data Sheet.
     */

    HWREG_SET_BITS( pReg->FPU_FPCC, ( FPCC_LSPEN | FPCC_ASPEN ) );
}


/**
 * Disables the stacking of floating-point registers
 * when an interrupt is handled.
 */
void fpu_disableStacking(void)
{
    /*
     * LSPEN and ASPEN flags of the FPCC register are cleared.
     * For more details, see page 196 of the Data Sheet.
     */

    HWREG_CLEAR_BITS( pReg->FPU_FPCC, ( FPCC_LSPEN | FPCC_ASPEN ) );
}


/**
 * Selects the format of half-precision floating-point values.
 *
 * The FPU supports either the IEEE 754 format or the alternative
 * Cortex-M format that has a larger range but does not have a way to
 * encode infinity (positive or negative) or NaN (quiet or signaling).
 *
 * Nothing is done if 'mode' has an invalid value.
 *
 * @param mode - format for half-precision FP value (any value of FpuHalfPrecisionMode)
 */
void fpu_setHalfPrecisionMode(FpuHalfPrecisionMode mode)
{
    /*
     * Half precision FP mode is adjusted by setting
     * or clearing the AHP flag of the FPDSC register.
     * For more details, see page 199 of the Data Sheet.
     */

    switch (mode)
    {
    case FPUHPM_IEEE:
        HWREG_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_AHP );
        break;

    case FPUHPM_ALTERNATIVE:
        HWREG_SET_BITS( pReg->FPU_FPDSC, FPDSC_AHP );
        break;
    };
}


/**
 * Selects the NaN mode.
 *
 * The FPU supports either propagation of NaNs or
 * returning the default NaN.
 *
 * Nothing is done if 'mode' has an invalid value.
 *
 * @param mode - the mode for NaN results (any value of FpuNanMode)
 */
void fpu_setNanMode(FpuNanMode mode)
{
    /*
     * NaN mode is adjusted by setting
     * or clearing the DN flag of the FPDSC register.
     * For more details, see page 199 of the Data Sheet.
     */

    switch (mode)
    {
    case FPU_NAN_PROPAGATE:
        HWREG_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_DN );
        break;

    case FPU_NAN_DEFAULT:
        HWREG_SET_BITS( pReg->FPU_FPDSC, FPDSC_DN );
        break;
    };
}


/**
 * Enables or disables the flush-to-zero mode of the
 * floating-point unit. When disabled, the FPU
 * is fully IEEE 754 compliant. When enabled, values
 * close to zero are treated as zero, thus greatly improving
 * the execution speed at the expense of some accuracy and
 * IEEE compliance.
 *
 * @param fz - 'true' to enable flushing to 0, 'false' to disable it
 */
void fpu_setFlushToZero(bool fz)
{
    /*
     * Flushing-to-zero is adjusted by setting
     * or clearing the FZ flag of the FPDSC register.
     * For more details, see page 199 of the Data Sheet.
     */

    if ( true == fz )
    {
        HWREG_SET_BITS( pReg->FPU_FPDSC, FPDSC_FZ );
    }
    else
    {
        HWREG_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_FZ );
    }
}


/**
 * Selects the rounding mode for floating-point results.
 *
 * The following rounding modes are supported by the FPU:
 * - rounding toward the nearest value
 * - rounding toward positive infinity
 * - rounding toward negative infinity
 * - rounding toward zero
 *
 * Nothing is done if 'mode' has an invalid value.
 *
 * @param mode - rounding mode (any value of FpuRMode)
 */
void fpu_setRoundingMode(FpuRMode mode)
{
    /*
     * Rounding mode is adjusted by appropriately setting
     * or clearing the RMODE bits of the FPDSC register.
     * For more details, see page 199 of the Data Sheet.
     */

    switch (mode)
    {
    case FPU_RMODE_RN:
        HWREG_SET_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_RMODE_RN, FPDSC_RMODE_MASK );
        break;

    case FPU_RMODE_RP:
        HWREG_SET_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_RMODE_RP, FPDSC_RMODE_MASK );
        break;

    case FPU_RMODE_RM:
        HWREG_SET_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_RMODE_RM, FPDSC_RMODE_MASK );
        break;

    case FPU_RMODE_RZ:
        HWREG_SET_CLEAR_BITS( pReg->FPU_FPDSC, FPDSC_RMODE_RZ, FPDSC_RMODE_MASK );
        break;
    };
}
