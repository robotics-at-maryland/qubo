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
 * Implementation of the board's Nested Vectored Interrupt
 * Controller (NVIC) functionality.
 *
 * For more info about the NVIC, see pp. 141 - 156 of:
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>

#include "bsp.h"
#include "nvic.h"
#include "regutil.h"


/*
 * 32-bit Registers of the NVIC,
 * relative to the first relevant NVIC register of
 * the peripheral controller.
 *
 * See pages 141 - 156 of the Data Sheet.
 *
 * Note: instead of providing separate fields for
 * registers ENx, DISx, PENDx, UNPENDx, ACTIVEx,and PRIx,
 * it is much more convenient to provide arrays of appropriate
 * sizes instead.
 */
typedef struct _TM4C123G_NVIC_REGS
{
    uint32_t NVIC_EN[5];                /* Interrupt 0-138 Set Enable */
    const uint32_t Reserved1[27];       /* reserved */
    uint32_t NVIC_DIS[5];               /* Interrupt 0-138 Clear Enable */
    const uint32_t Reserved2[27];       /* reserved */
    uint32_t NVIC_PEND[5];              /* Interrupt 0-138 Set Pending */
    const uint32_t Reserved3[27];       /* reserved */
    uint32_t NVIC_UNPEND[5];            /* Interrupt 0-138 Clear Pending */
    const uint32_t Reserved4[27];       /* reserved */
    const uint32_t NVIC_ACTIVE[5];      /* Interrupt 0-138 Active Bit, read only */
    const uint32_t Reserved5[59];       /* reserved */
    uint32_t NVIC_PRI[35];              /* Interrupt 0-138 Priority */
    const uint32_t Reserved6[669];      /* reserved */
    uint32_t NVIC_SWTRIG;               /* Software Trigger Interrupt, write only */
} TM4C123G_NVIC_REGS;


static volatile TM4C123G_NVIC_REGS* const pReg =
   (volatile TM4C123G_NVIC_REGS* const) ( BSP_NVIC_BASE_ADDRESS );


/* Number of bits per register: */
#define BITS_PER_REG          ( 32 )

/* Total number of all IRQ requests, supported by the NVIC:*/
#define NR_INTRS              ( 139 )

/* Convenience bit mask for setting the NVIC_PRI registers:*/
#define INTR_PRI_MASK         MAX_PRIORITY

/* Number of IRQ priorities, supported by a single NVIC_PRI register: */
#define INTRS_PER_PRI_REG     ( 4 )



/**
 * Unmasks the selected IRQ request.
 *
 * Nothing is done if 'irq' is greater than 138.
 *
 * @param irq - IRQ number to be enabled (between 0 and 138)
 */
void nvic_enableInterrupt(uint8_t irq)
{
    /*
     * Set the appropriate bit in the appropriate NVIC_EN register.
     * For more details, see pp. 142 - 143 of the Data Sheet.
     */

    if ( irq < NR_INTRS )
    {
        HWREG_SET_SINGLE_BIT( pReg->NVIC_EN[ irq / BITS_PER_REG ], irq % BITS_PER_REG);
    }
}


/**
 * Masks the selected IRQ request.
 *
 * Nothing is done if 'irq' is greater than 138.
 *
 * @param irq - IRQ number to be disabled (between 0 and 138)
 */
void nvic_disableInterrupt(uint8_t irq)
{
    /*
     * Clear the appropriate bit in the appropriate NVIC_EN register.
     * For more details, see pp. 142 - 143 of the Data Sheet.
     */

    if ( irq < NR_INTRS )
    {
        HWREG_SET_SINGLE_BIT( pReg->NVIC_DIS[ irq / BITS_PER_REG ], irq % BITS_PER_REG);
    }
}


/*
 * A convenience inline function that returns the correct
 * shift position of bits that handle the (IRQ%4)'s
 * priority.
 *
 *   shift = 8 * mod + 5
 *
 * @param mod - remainder of 4 (between 0 and 3)
 *
 * @return NVIC_PRI register's shift for IRQ%4 priority handling bits
 */
static inline uint8_t __priShift(uint8_t mod)
{
    /*
     * See page 152 of the Data Sheet.
     */
    return ( mod * 8 + 5);
}


/**
 * Sets priority for the selected IRQ request.
 *
 * Nothing is done if 'irq' is greater than 138 or
 * 'pri' is greater than 7.
 *
 * @param irq -IRQ request whose priority will be set (between 0 and 138)
 * @param pri - priority level to be set (between 0 and 7)
 */
void nvic_setPriority(uint8_t irq, uint8_t pri)
{
    /*
     * For more details, see pp. 152 - 155 of the Data Sheet.
     */

    const uint8_t shift = __priShift(irq % INTRS_PER_PRI_REG);

    if ( irq < NR_INTRS && pri <= INTR_PRI_MASK )
    {
        HWREG_SET_CLEAR_BITS( pReg->NVIC_PRI[irq / INTRS_PER_PRI_REG],
                pri << shift,
                INTR_PRI_MASK << shift );
    }
}


/**
 * Returns priority of the selected IRQ request.
 *
 * Nothing is done if 'irq' is greater than 138.
 *
 * @param irq -IRQ request whose priority will be set (between 0 and 138)
 *
 * @return priority of the selected IRQ
 */
uint8_t nvic_getPriority(uint8_t irq)
{
    /*
     * For more details, see pp. 152 - 155 of the Data Sheet.
     */

    const uint8_t shift = __priShift(irq % INTRS_PER_PRI_REG);

    return
        ( irq < NR_INTRS ?
          HWREG_READ_BITS( pReg->NVIC_PRI[irq / INTRS_PER_PRI_REG], INTR_PRI_MASK << shift) >> shift :
          0 );
}
