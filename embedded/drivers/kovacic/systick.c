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
 * Implementations of the System Timer (SysTick) functionality.
 *
 * For more information about the SysTick, see page 123,
 * for details about its registers, see pp. 137 - 141 of the
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
#include "scb.h"


/*
 * 32-bit registers of the SysTick controller, relative of the
 * first SysTick related register of the Peripheral Controller.
 * 
 * See page 134 of the Data Sheet.
 */
typedef struct _TM4C123G_SYSTICK_REGS
{
    uint32_t SYSTICK_STCTRL;            /* SysTick Control and Status Register */
    uint32_t SYSTICK_STRELOAD;          /* SysTick Reload Value Register */
    uint32_t SYSTICK_STCURRENT;         /* SysTick Current Value Register */
} TM4C123G_SYSTICK_REGS;


/* A pointer to the SysTick "base" address: */
static volatile TM4C123G_SYSTICK_REGS* const pReg =
   (volatile TM4C123G_SYSTICK_REGS* const) ( BSP_SYSTICK_BASE_ADDRESS );


/*
 * Bit masks of the Control and Status Register (STCTRL).
 * For a detailed description of each bit, see pp. 138 - 139
 * of the Data Sheet.
 */
#define ENABLE_MASK          ( 0x00000001 )
#define INTR_MASK            ( 0x00000002 )
#define CLKSRC_MASK          ( 0x00000004 )
#define COUNT_MASK           ( 0x00010000 )

/* A convenience 24-bit mask, useful for the reload operation: */
#define B24BIT_MASK          ( 0x00FFFFFF )


/**
 * Stops the SysTick timer.
 */
void systick_disable(void)
{
    HWREG_CLEAR_BITS( pReg->SYSTICK_STCTRL, ENABLE_MASK );
}


/**
 * Starts the SysTick timer.
 */
void systick_enable(void)
{
    HWREG_SET_BITS( pReg->SYSTICK_STCTRL, ENABLE_MASK );
}


/**
 * Sets the SysTick's source clock.
 * The SysTick runs either on the system clock or on the
 * precision internal oscillator (PIOSC), whose frequency
 * is divided by 4.
 *
 * @param systemClock - 'true' for the system clock; 'false' for PIOSC/4
 */
void systick_setSource(bool systemClock)
{
    if ( true == systemClock)
    {
        HWREG_SET_BITS( pReg->SYSTICK_STCTRL, CLKSRC_MASK );
    }
    else
    {
        HWREG_CLEAR_BITS( pReg->SYSTICK_STCTRL, CLKSRC_MASK );
    }
}


/**
 * Has the counter already wrapped (reached 0)?
 * Note that the count flag is automatically cleared
 * after its status has been read.
 *
 * @return 'true' if the counter reached 0; 'false' if not
 */
bool systick_countSet(void)
{
    return ( HWREG_READ_BITS( pReg->SYSTICK_STCTRL, COUNT_MASK ) ? true : false );
}


/**
 * Sets the starting value of the counter.
 * The counter is also reset to this value when it reaches 0.
 *
 * Note that the reload value must be a 24-bit number.
 *
 * @param value - reload value (the most significant 8 bits will be discarded)
 */
void systick_setReload(uint32_t value)
{
    if ( (value & B24BIT_MASK) > 0 )
    {
        HWREG_SET_CLEAR_BITS( pReg->SYSTICK_STRELOAD, value, B24BIT_MASK );
    }
}


/**
 * Resets (clears) the counter state to the reload value.
 */
void systick_clear(void)
{
    /* writing any value to dedicated bits of the STCURRENT will clear the register and count. */
    HWREG_SET_BITS( pReg->SYSTICK_STCURRENT, B24BIT_MASK );
}


/**
 * @return current value of the SysTick counter
 */
uint32_t systick_getCurrentValue(void)
{
    return ( HWREG_READ_BITS( pReg->SYSTICK_STCURRENT, B24BIT_MASK ) );
}


/**
 * Enables triggering an interrupt each time the counter reaches 0.
 */
void systick_enableInterrupt(void)
{
    HWREG_SET_BITS( pReg->SYSTICK_STCTRL, INTR_MASK );
}


/**
 * Disables SysTick interrupt triggering.
 */
void systick_disableInterrupt(void)
{
    HWREG_CLEAR_BITS( pReg->SYSTICK_STCTRL, INTR_MASK );
}


/**
 * A convenience dummy function for acknowledgment/clearing of a
 * SysTick triggered interrupt.
 *
 * @note SysTick is the only interrupt with automatic acknowledgment
 *       (http://users.ece.utexas.edu/~valvano/Volume1/E-Book/C12_Interrupts.htm),
 *       hence the implementation of this function is empty and there
 *       is indeed no need to ever call it.
 *
 * @note If a high priority ISR wants to clear a pending SysTick's flag,
 *       it should call scb_unpendSysTickIntr() instead.
 */
void systick_clearInterrupt(void)
{
    /* empty function */
}


/**
 * Sets priority of the SysTick triggered interrupts.
 *
 * Nothing is done if 'pri' is greater than 7.
 *
 * @param pri - priority level of SysTick generated interrupts (between 0 and 7)
 */
void systick_setPriority(uint8_t pri)
{
    /* The SysTick's priority is actually set in the SCB */
    scb_setSysTickPriority(pri);
}


/**
 * Initial configuration of the SysTick.
 *
 * System clock is set as the clock source, counter reload
 * value is set to the desired value and the counter is cleared.
 *
 * After configuration, the SysTick is disabled (not running)
 * and triggering of interrupts is disabled.
 */
void systick_config(uint32_t reload)
{
    /* Stop the SysTick if it is running */
    systick_disable();

    /*
     * According to the Data Sheet, page 138, the clock source
     * should be set by default to the system clock, but apparently
     * it isn't, so it will be set using this command.
     */
    systick_setSource(true);

    /* Set the counter to the reload value */
    systick_setReload(reload);

    /* And finally clear the counter */
    systick_clear();

    /* Initially triggering of interrupts will be disabled */
    systick_disableInterrupt();
}
