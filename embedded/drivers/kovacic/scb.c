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
 * Implementation of the board's System Control Block
 * (SCB) functionality that provides system implementation
 * information and system control, including configuration,
 * control, and reports of system exceptions.
 *
 *
 * For more info about the SCB, see pp. 156 - 185 of:
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */

#include <stdint.h>

#include "bsp.h"
#include "regutil.h"



/*
 * 32-bit Registers of the SCB,
 * relative to the first relevant SCB register of
 * the peripheral controller.
 *
 * See pages 156 - 185 of the Data Sheet.
 */
typedef struct _TM4C123G_SCB_REGS
{
    uint32_t SCB_ACTLR;                /* Auxiliary Control */
    const uint32_t Reserved1[829];     /* reserved */
    const uint32_t SCB_CPUID;          /* CPU ID Base */
    uint32_t SCB_INTCTRL;              /* Interrupt Control and State */
    uint32_t SCB_VTABLE;               /* Vector Table Offset */
    uint32_t SCB_APINT;                /* Application Interrupt and Reset Control */
    uint32_t SCB_SYSCTRL;              /* System Control */
    uint32_t SCB_CFGCTRL;              /* Configuration and Control */
    uint32_t SCB_SYSPRI1;              /* System Handler Priority 1 */
    uint32_t SCB_SYSPRI2;              /* System Handler Priority 2 */
    uint32_t SCB_SYSPRI3;              /* System Handler Priority 3 */
    uint32_t SCB_SYSHNDCTRL;           /* System Handler Control and State */
    uint32_t SCB_FAULTSTAT;            /* Configurable Fault Status */
    uint32_t SCB_HFAULTSTAT;           /* Hard Fault Status */
    const uint32_t Reserved2;          /* reserved */
    uint32_t SCB_MMADDR;               /* Memory Management Fault Address */
    uint32_t SCB_FAULTADDR;            /* Bus Fault Address */
} TM4C123G_SCB_REGS;


static volatile TM4C123G_SCB_REGS* const pReg =
   (volatile TM4C123G_SCB_REGS* const) ( BSP_SCB_BASE_ADDRESS );


/* Convenience bit mask for setting of SysTick's interrupt priority */
#define PRI_MASK                        ( 0x00000007 )

/*
 * Bit mask with relevant bits of the SYSPRI3 register that
 * handle SysTick's interrupt priority
 */
#define SYSTICK_PRI_MASK                ( 0xE0000000 )

/*
 * Convenience offset of the SYSPRI3 register where SysTick's
 * priority bits start.
 */
#define SYSTICK_PRI_OFFSET              ( 29 )

/*
 * Bit mask with relevant bits of the SYSPRI3 register that
 * handle PendSV interrupt priority
 */
#define PENDSV_PRI_MASK                 ( 0x00E00000 )

/*
 * Convenience offset of the SYSPRI3 register where PendSV
 * priority bits start.
 */
#define PENDSV_PRI_OFFSET               ( 21 )

/*
 * VECTKEY value and its bit mask.
 */
#define APINT_VECTKEY                   ( 0x05FA0000 )
#define APINT_VECTKEY_MASK              ( 0xFFFF0000 )

/* Bit mask for SYSRESREQ flag */
#define APINT_SYSRESREQ                 ( 0x00000004 )

/* Bit mask for VECACT status bits of the INTCTRL register: */
#define INTCTRL_VECACT_MASK             ( 0x000000FF )

/* Flags for setting and clearing PendSV pending */
#define INTCTRL_PENDSV_FLAG             ( 0x10000000 )
#define INTCTRL_UNPENDSV_FLAG           ( 0x08000000 )

/* Flags for setting and clearing SysTick pending */
#define INTCTRL_PENDSTSET_FLAG          ( 0x04000000 )
#define INTCTRL_PENDSTCLR_FLAG          ( 0x02000000 )


/**
 * Sets priority of SysTick generated interrupt requests
 *
 * Nothing is done if 'pri' is greater than 7.
 *
 * @param pri - priority of SysTick generated interrupts (between 0 and 7)
 */
void scb_setSysTickPriority(uint8_t pri)
{
    /*
     * For more details, see page 172 of the Data Sheet.
     */

    if ( pri <= PRI_MASK )
    {
        HWREG_SET_CLEAR_BITS( pReg->SCB_SYSPRI3, pri<<SYSTICK_PRI_OFFSET, SYSTICK_PRI_MASK );
    }
}


/**
 * Sets priority of PendSV interrupt requests.
 *
 * Nothing is done if 'pri' is greater than 7.
 *
 * @param pri - priority of PendSV interrupts (between 0 and 7)
 */
void scb_setPendSvPriority(uint8_t pri)
{
    /*
     * For more details, see page 172 of the Data Sheet.
     */

    if ( pri <= PRI_MASK )
    {
        HWREG_SET_CLEAR_BITS( pReg->SCB_SYSPRI3, pri<<PENDSV_PRI_OFFSET, PENDSV_PRI_MASK );
    }
}


/*
 * Triggers a PendSV exception.
 *
 * This is a user triggered exception, convenient to notify
 * the scheduler to switch a context.
 */
void scb_triggerPendSv(void)
{
    /*
     * PendSV pending is triggered via the PENDSV flag
     * of the INTCTRL register.
     *
     * For more details, see page 160 of the Data Sheet.
     */

    HWREG_SET_BITS( pReg->SCB_INTCTRL, INTCTRL_PENDSV_FLAG );
}


/*
 * Clears the PendSV pending status.
 */
void scb_clearPendSv(void)
{
    /*
     * PendSV's pending state is removed by setting of the
     * INTCTRL register's PENDSV flag.
     *
     * For more details, see page 161 of the Data Sheet.
     */

    HWREG_SET_BITS( pReg->SCB_INTCTRL, INTCTRL_UNPENDSV_FLAG );
}


/**
 * Resets the CPU and on-chip peripherals.
 */
void scb_reset(void)
{
    /*
     * As evident from page 217 of the Data Sheet,
     * software initiated reset can be performed by
     * setting the SYSRESREQ flag of the SCB's APINT
     * register.
     *
     * For more info about the APINT register, see
     * pp. 164 - 165 of the Data Sheet.
     * In order to modify its bits, the upper 16 bits
     * must be set to 0x05FA.
     */

    pReg->SCB_APINT = APINT_VECTKEY | APINT_SYSRESREQ;

    /* just in case */
    for ( ; ; );
}


/**
 * Set SysTick pending status.
 */
void scb_pendSysTickIntr(void)
{
    /*
     * SysTick pending is triggered via the PENDSTSET flag
     * of the INTCTRL register.
     *
     * For more details, see page 161 of the Data Sheet.
     */

    HWREG_SET_BITS( pReg->SCB_INTCTRL, INTCTRL_PENDSTSET_FLAG );
}


/**
 * Clear SysTick pending status.
 *
 * @note The ISR handling SysTick exception does not need to
 *       call this function as the trigger flag is automatically
 *       cleared when the SysTick exception ISR starts:
 *       http://users.ece.utexas.edu/~valvano/Volume1/E-Book/C12_Interrupts.htm
 */
void scb_unpendSysTickIntr(void)
{
    /*
     * SysTick's pending state is removed by setting of the
     * INTCTRL register's PENDSTCLR flag.
     *
     * For more details, see page 161 of the Data Sheet.
     */

    HWREG_SET_BITS( pReg->SCB_INTCTRL, INTCTRL_PENDSTCLR_FLAG );
}


/**
 * @return active exception number (if in Handler mode) or 0 (if in Thread mode)
 */
uint8_t scb_activeException(void)
{
    return (uint8_t) HWREG_READ_BITS( pReg->SCB_INTCTRL, INTCTRL_VECACT_MASK );
}
