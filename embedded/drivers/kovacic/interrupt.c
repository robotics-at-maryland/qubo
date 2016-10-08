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
 * Implementation of functions that enable or disable activation
 * of exceptions/interrupts and define the minimum priority for
 * exception processing.
 *
 * Note: this can only be achieved by modifying bits of system
 * registers (e.g. PRIMASK, BASEPRI) that are not mapped to the
 * memory, hence the functions must be at least partially
 * implemented in assembler.
 *
 * For more details, see
 * Cortex-M4 Devices, Generic User Guide (DUI553A):
 * http://infocenter.arm.com/help/topic/com.arm.doc.dui0553a/DUI0553A_cortex_m4_dgug.pdf
 *
 * and
 *
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */

#include <stdint.h>

#include "nvic.h"


/*
* A convenience set of macros that allow usage
* of #define'd constants in inline assembler directives.
*
* STR(arg) should always be called inside __asm().
* Two macros ensure that 'arg' will be stringized into "arg"
* first, then it will be expanded according #define macros,
* finally the preprocessor will concatenate strings.
*
* For more details, see:
* http://stackoverflow.com/questions/10495752/can-a-macro-be-used-in-inline-asm
*/
#define STRINGIZE(arg)        #arg
#define STR(arg)              STRINGIZE(arg)


/* Convenience constants used in inline assembler directives:*/

/* Bit mask of the BASEPRI register with relevant bits: */
#define BASEPRI_MASK          0x000000E0
/* Position of the least significant bit that determines priority: */
#define BASEPRI_OFFSET        5



/**
 * Enables processing of all exceptions and interrupts (with
 * sufficient priority).
 */
void intr_enableInterrupts(void)
{
    /*
     * The following assembler instruction directly clears
     * the special purpose register PRIMASK and thus enables
     * interrupts and configurable fault handlers.
     *
     * See the Data Sheet, page 85, for more details about
     * the PRIMASK register.
     *
     * See DUI553A, page 3-159, for more details.
     */

    __asm volatile("     CPSIE  i");
}


/**
 * Disables processing of all interrupts and exceptions with
 * programmable priority. Reset, non-maskable interrupt (NMI)
 * and hard fault exception will remain enabled.
 */
void intr_disableInterrupts(void)
{
    /*
     * The following assembler instruction directly sets
     * the special purpose register PRIMASK and thus disables
     * interrupts and configurable fault handlers.
     *
     * See the Data Sheet, page 85, for more details about
     * the PRIMASK register.
     *
     * See DUI553A, page 3-159, for more details.
     */

    __asm volatile("     CPSID  i");
}


/**
 * Enables processing of all exceptions and fault handlers.
 */
void intr_enableException(void)
{
    /*
     * The following assembler instruction directly clears
     * the special purpose register FAULTMASK and thus enables
     * activation of all exceptions and interrupts.
     *
     * See the Data Sheet, page 86, for more details about
     * the FAULTMASK register.
     *
     * See DUI553A, page 3-159, for more details.
     */

    __asm volatile("     CPSIE  f");

}


/**
 * Disables processing of all exceptions and fault handlers
 * except non-maskable interrupt (NMI).
 */
void intr_disableException(void)
{
    /*
     * The following assembler instruction directly sets
     * the special purpose register FAULTMASK and thus disables
     * all exceptions except NMI (Non-Maskable Interrupt).
     *
     * See the Data Sheet, page 86, for more details about
     * the FAULTMASK register.
     *
     * See DUI553A, page 3-159, for more details.
     */

    __asm volatile("     CPSID  f");

}


/**
 * Sets the minimum priority for exception/interrupt
 * processing. Processing of all exceptions with the same
 * or lower priority (note that higher 'pri' means lower
 * priority!) will be disabled. If 0 is passed, all
 * exceptions will be processed.
 *
 * Nothing is done if 'pri' is greater than 7.
 *
 * @param pri - minimum priority (between 0 and 7)
 */
void intr_setBasePriority(uint8_t pri)
{
    /*
     * The function is entirely implemented in assembler.
     * Note that most compilers will pass the first (and in
     * this case the only) function's input argument to R0.
     *
     * The BASEPRI register is copied into a general purpose register
     * R4 using the MRS instruction, relevant bits are first cleared
     * and then appropriately set, then the R4 is copied back to BASEPRI
     * using the MSR instruction.
     *
     * For more details about the BASEPRI register, see the
     * Data Sheet, page 87.
     */

    __asm volatile("     MOV r3, r0                        ");  /* r3 = r0                    */
    __asm volatile("     CMP r3, #" STR(MAX_PRIORITY)       );  /* if (r3>7) ...              */
    __asm volatile("     BHI skip_pri_set                  ");  /* ... then goto skip_pri_set */
    __asm volatile("     LSL r3, r3, #" STR(BASEPRI_OFFSET) );  /* r3 <<= 5                   */
    __asm volatile("     MRS r4, basepri                   ");  /* r4 = basepri               */
    __asm volatile("     BIC r4, r4, #" STR(BASEPRI_MASK)   );  /* r4 &= ~0x000000E0          */
    __asm volatile("     ORR r4, r4, r3                    ");  /* r4 |= r3                   */
    __asm volatile("     MSR basepri, r4                   ");  /* basepri = r4               */
    __asm volatile("skip_pri_set:                          ");

    /*
     * Suppress the warning, that "'pri' is never used".
     * Actually it is implicitly used (passed to the function as r0)
     * in the assembler code above.
     */
    (void) pri;
}
