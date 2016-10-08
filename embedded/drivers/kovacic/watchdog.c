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
 * Implementation of watchdog functionality.
 * Both watchdog timers are supported.
 *
 * All functions also take care of the time gap, necessary
 * write access of the Watchdog 1's registers are performed.
 *
 * For more info about the watchdog timers,
 * see pp. 775 - 799 of:
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * Important workarounds for hardware bugs in the watchdog 1
 * are also discussed in pp. 63 - 69 of
 * Tiva(TM) C Series TM4C123x Microcontrollers Silicon Revisions 6 and 7,
 *     Silicon Errata,
 * available at:
 * http://www.ti.com/lit/er/spmz849c/spmz849c.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "bsp.h"
#include "regutil.h"
#include "watchdog.h"
#include "sysctl.h"
#include "nvic.h"



/*
 * Convenience bit masks for setting of the WDCTL register
 */
#define CTL_WRC                  ( 0x80000000 )
#define CTL_INTTYPE              ( 0x00000004 )
#define CTL_RESEN                ( 0x00000002 )
#define CTL_INTEN                ( 0x00000001 )

/* Watchdog Stall Enable flag */
#define TST_STALL                ( 0x00000100 )

/* Convenience bit mask for the WDTMIS register */
#define MIS_MASK                 ( 0x00000001 )

/* A value to unlock watchdog's registers */
#define UNLOCK_VAL               ( 0x1ACCE551 )

/*
 * 32-bit Registers of individual watchdog timers,
 * relative to the controller's base address:
 * See page 778 of the Data Sheet.
 */
typedef struct _TM4C123G_WATCHDOG_REGS
{
    uint32_t WD_LOAD;                   /* Watchdog Load */
    const uint32_t WD_VALUE;            /* Watchdog Value, read only */
    uint32_t WD_CTL;                    /* Watchdog Control */
    uint32_t WD_ICR;                    /* Watchdog Interrupt Clear, write only */
    const uint32_t WD_RIS;              /* Watchdog Raw Interrupt Status, read only */
    const uint32_t WD_MIS;              /* Watchdog Masked Interrupt Status, read only */
    const uint32_t Reserved1[256];      /* reserved */
    uint32_t WD_TEST;                   /* Watchdog Test */
    const uint32_t Reserved2[505];      /* reserved */
    uint32_t WD_LOCK;                   /* Watchdog Lock */
    const uint32_t Reserved3[243];      /* reserved */
    const uint32_t WD_PeriphID4;        /* Watchdog Peripheral Identification 4, read only */
    const uint32_t WD_PeriphID5;        /* Watchdog Peripheral Identification 5, read only */
    const uint32_t WD_PeriphID6;        /* Watchdog Peripheral Identification 6, read only */
    const uint32_t WD_PeriphID7;        /* Watchdog Peripheral Identification 7, read only */
    const uint32_t WD_PeriphID0;        /* Watchdog Peripheral Identification 0, read only */
    const uint32_t WD_PeriphID1;        /* Watchdog Peripheral Identification 1, read only */
    const uint32_t WD_PeriphID2;        /* Watchdog Peripheral Identification 2, read only */
    const uint32_t WD_PeriphID3;        /* Watchdog Peripheral Identification 3, read only */
    const uint32_t WD_PCellID0;         /* Watchdog PrimeCell Identification 0, read only */
    const uint32_t WD_PCellID1;         /* Watchdog PrimeCell Identification 1, read only */
    const uint32_t WD_PCellID2;         /* Watchdog PrimeCell Identification 2, read only */
    const uint32_t WD_PCellID3;         /* Watchdog PrimeCell Identification 3, read only */
} TM4C123G_WATCHDOG_REGS;


/* --------------------------------------------------- */
#define GEN_CAST_ADDR(ADDR)         (volatile TM4C123G_WATCHDOG_REGS* const) (ADDR),

static volatile TM4C123G_WATCHDOG_REGS* const pReg[ BSP_NR_WATCHDOGS ] =
{
    BSP_WATCHDOG_BASE_ADDRESSES( GEN_CAST_ADDR )
};

#undef GEN_CAST_ADDR
/* --------------------------------------------------- */


/* Settings of each watchdog timer: */
struct WdSettings_t
{
    uint32_t loadVal;               /* load value */
    WatchdogException exType;       /* exception mode */
    bool enabled;                   /* Sysctl enable status */
    WatchdogIntHandler_t intrIsr;   /* interrupt handler */
};

static struct WdSettings_t __wdSettings[ BSP_NR_WATCHDOGS ];


/* Declared in sysctl.c */
extern int8_t sysctl_mcu_revision;


/*
 * The Watchdog 1 is clocked by an independent source and
 * its registers must be written with a time gap between
 * accesses. If the WRC bit of the WDTCTL register is set to 1,
 * this indicates that the timing gap has elapsed. There are no
 * such restrictions for the Watchdog 0.
 *
 * See pages 777 and 781 of the Data Shhet for more details.
 *
 * @param wd - watchdog timer
 */
static inline void __waitWd1(uint8_t wd)
{
    if ( 1 == wd )
    {
        while ( 0 == HWREG_READ_BITS(pReg[wd]->WD_CTL, CTL_WRC) );
    }
}


/**
 * Enables the selected watchdog timer at the System Controller.
 * When the watchdog timer is enabled, it is provided a clock and
 * access to its registers is allowed.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer to be enabled (between 0 and 1)
 */
void wd_enableWd(uint8_t wd)
{
    if ( wd < BSP_NR_WATCHDOGS )
    {
        sysctl_enableWatchdog(wd);
        __wdSettings[wd].enabled = true;
    }
}


/**
 * Disables the selected watchdog timer at the System Controller.
 * When the watchdog timer is disabled, it is disconnected from the clock
 * and access to its registers is not allowed. Any attempt of
 * accessing a disabled watchdog timer may result in a bus fault.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer to be enabled (between 0 and 1)
 */
void wd_disableWd(uint8_t wd)
{
    if ( wd < BSP_NR_WATCHDOGS )
    {
        sysctl_disableWatchdog(wd);
        __wdSettings[wd].enabled = false;
    }
}


/*
 * Configures an exception that is triggered when the watchdog's
 * counter reaches 0.
 *
 * Nothing is done if 'wd' is greater than 1 or
 * 'ex' is invalid.
 *
 * @param wd - watchdog timer to be configured (between 0 and 1)
 * @param ex - exception type (any valid value of WatchdogException)
 */
static void __wd_setExceptionType(uint8_t wd, WatchdogException ex)
{
    /*
     * Exception type is configured via relevant bits
     * of the WDTCTL register.
     * See the Data Sheet, pp. 781 - 782, for more details.
     */

    if ( wd >= BSP_NR_WATCHDOGS )
    {
        return;
    }

    __wdSettings[wd].exType = ex;

    /* Clear the RESEN flag. It will be reenabled if necessary */
    HWREG_CLEAR_BITS( pReg[wd]->WD_CTL, CTL_RESEN );
    __waitWd1(wd);

    switch (ex)
    {
    case WDEX_NMI:
        HWREG_SET_BITS( pReg[wd]->WD_CTL, CTL_INTTYPE );
        break;

    case WDEX_IRQ:
        HWREG_CLEAR_BITS( pReg[wd]->WD_CTL, CTL_INTTYPE );
        break;

    case WDEX_NMI_RESET:
        HWREG_SET_BITS( pReg[wd]->WD_CTL, CTL_INTTYPE );
        __waitWd1(wd);
        HWREG_SET_BITS( pReg[wd]->WD_CTL, CTL_RESEN );
        break;

    case WDEX_IRQ_RESET:
        HWREG_CLEAR_BITS( pReg[wd]->WD_CTL, CTL_INTTYPE );
        __waitWd1(wd);
        HWREG_SET_BITS( pReg[wd]->WD_CTL, CTL_RESEN );
        break;

    }

    __waitWd1(wd);
}


/*
 * Assigns the selected register a 32-bit reload value.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer to be loaded (between 0 and 1)
 * @param ex - exception type
 */
static void __wd_setLoadValue(uint8_t wd, uint32_t value)
{
    /*
     * See page 779 of the Data Sheet for more details.
     */

    if ( wd < BSP_NR_WATCHDOGS )
    {
        /* store the value for reusing it by wd_reload() */
        __wdSettings[wd].loadVal = value;

        /*
         * There is a hardware bug in both watchdog timers.
         * If the STALL bit of the WDTTEST register is set,
         * the WTDLOAD register cannot be changed.
         * See page 68 of the Silicon Errata for more details.
         *
         * For that reason the STALL bit will be cleared first
         * and set again later.
         */
        HWREG_CLEAR_BITS( pReg[wd]->WD_TEST, TST_STALL );
        __waitWd1(wd);

        pReg[wd]->WD_LOAD = value;
        __waitWd1(wd);

        /* Set the STALL bit again. */
        HWREG_SET_BITS( pReg[wd]->WD_TEST, TST_STALL );
        __waitWd1(wd);
    }
}


/**
 * Starts the selected watchdog timer and enables
 * triggering of interrupts.
 *
 * @note Once the watchdog has been start it is
 * only possible to stop it by reseting it, either
 * via wd_reset() or by a hardware reset.
 * See page 782 of the Data Sheet for more details.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer (between 0 and 1)
 */
void  wd_start(uint8_t wd)
{
    /*
     * A watchdog is enabled by the INTEN bit
     * of the WDTCTL register.
     * See the Data Sheet, page 782 for more details.
     */
    if ( wd < BSP_NR_WATCHDOGS )
    {

        HWREG_SET_BITS( pReg[wd]->WD_CTL, CTL_INTEN );
        __waitWd1(wd);

        /* and leave registers unlocked: */
        pReg[wd]->WD_LOCK = UNLOCK_VAL;
        __waitWd1(wd);
    }
}


/**
 * Resets the selected watchdog module.
 * This is the only way to disable it once it
 * has been enabled.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer (between 0 and 1)
 */
void wd_reset(uint8_t wd)
{
    if ( wd < BSP_NR_WATCHDOGS )
    {
        sysctl_resetWatchdog(wd);
    }
}


/**
 * Unmasks both watchdog timers' IRQ requests on NVIC.
 */
void wd_enableNvicIntr(void)
{
	nvic_enableInterrupt( BSP_WATCHDOG_IRQ );
}


/**
 * Masks both watchdog timers' IRQ requests on NVIC.
 */
void wd_disableNvicIntr(void)
{
	nvic_disableInterrupt( BSP_WATCHDOG_IRQ );
}


/**
 * Sets priority for the shared watchdog timers'
 * interrupt handler.
 *
 * Nothing is done if 'pri' is greater than 7.
 *
 * @param pri - priority level to be set (between 0 and 7)
 */
void wd_setIntPriority(uint8_t pri)
{
	if ( pri <= MAX_PRIORITY )
	{
        nvic_setPriority(BSP_WATCHDOG_IRQ, pri);
	}
}


/**
 * Clears interrupt status of the selected watchdog timer
 * and reloads the counter.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer (between 0 and 1)
 */
void wd_clearInterrupt(uint8_t wd)
{
    /*
     * Writing anything to the WDTICR register
     * will clear the interrupt.
     * For more details, see page 783 of the Data Sheet.
     */
    if ( wd < BSP_NR_WATCHDOGS )
    {
        __waitWd1(wd);
        pReg[wd]->WD_ICR = MIS_MASK;
        __waitWd1(wd);
    }
}


/**
 * Returns the current value of the selected watchdog timer.
 *
 * 0 is returned if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer (between 0 and 1)
 *
 * @return 32-bit current value of the watchdog timer
 */
uint32_t wd_getValue(uint8_t wd)
{
    /* The watchdog timer's current value is
     * stored in the WDTVALUE register.
     * See page 780 of the Data Sheet for more details.
     */

    return
        ( wd<BSP_NR_WATCHDOGS ? pReg[wd]->WD_VALUE : 0 );
}


/**
 * Registers a function that handles interrupt requests
 * triggered by the specified watchdog timer.
 *
 * Nothing is done if 'wd' is greater than 1.
 * NULL is a perfectly acceptable option for 'isr'.
 * In this case the interrupt processing routine will 
 * not do anything and will return immediately.
 *
 * @param wd - watchdog timer (between 0 and 1)
 * @param isr - address of the interrupt handling routine
 */
void wd_registerIntHandler(uint8_t wd, WatchdogIntHandler_t isr)
{
    if ( wd < BSP_NR_WATCHDOGS )
    {
        __wdSettings[wd].intrIsr = isr;
    }
}


/**
 * Unregisters the interrupt servicing routine for the specified
 * watchdog timer. If such a watchdog still triggers an interrupt,
 * the processing routine will not do anything and will return
 * immediately.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer (between 0 and 1)
 */
void wd_unregisterIntHandler(uint8_t wd)
{
    if ( wd < BSP_NR_WATCHDOGS )
    {
        __wdSettings[wd].intrIsr = NULL;
    }
}


/**
 * Configures the selected watchdog timer.
 * When this function completes, the watchdog
 * timer is not enabled yet, this must be done
 * separately by calling wd_start().
 *
 * The function does not modify the watcdog's
 * interrupt request handler, specified by
 * wd_registerIntHandler().
 *
 * Note that reset (if configured so) only occurs
 * after the second time out of a watchdog if the
 * previous time out's interrupt request has not
 * been cleared yet. See pages 217 - 218 and 775
 * of the Data Sheet for more details.
 *
 * Nothing is done if 'wd' is greater than 1 or
 * 'ex' is invalid.
 *
 * @param wd - watchdog timer to be configured (between 0 and 1)
 * @param ex - exception type (any valid value of WatchdogException)
 */
void wd_config(uint8_t wd, uint32_t loadValue, WatchdogException ex)
{
    if ( wd < BSP_NR_WATCHDOGS )
    {
        wd_enableWd(wd);

        /* set load value... */
        __wd_setLoadValue(wd, loadValue);

        /* ...configure exception handling... */
        __wd_setExceptionType(wd, ex);

        /* ...and enable WD interrupt on NVIC */
        wd_enableNvicIntr();
        
    }
}


/**
 * Reloads the selected watchdog's counter to the
 * load value, specified by wd_config().
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - watchdog timer (between 0 and 1)
 */
void wd_reload(uint8_t wd)
{
    /*
     * For more details about the WDTLOAD register,
     * see page 779 of the Data Sheet
     */

    if ( wd >= BSP_NR_WATCHDOGS )
    {
        return;
    }

    if ( 1==wd && (6==sysctl_mcu_revision || 7==sysctl_mcu_revision) )
    {
        /*
         * According to page 64 of the Silicon Errata,
         * reloading the WDTLOAD of the watchdog 1 will not
         * restart the counter. As a workaround it is
         * suggested to reset, reconfigure and restart
         * the watchdog.
         */

        wd_reset(wd);
        wd_config(wd, __wdSettings[wd].loadVal, __wdSettings[wd].exType);

        /* Set the STALL bit if necessary */
        HWREG_SET_BITS( pReg[wd]->WD_TEST, TST_STALL );
        __waitWd1(wd);

        wd_start(wd);
    }
    else
    {
        /*
         * At other watchdogs, no workaround is necessary.
         * Reloading the WDTLOAD register will also
         * restart the counter.
         * Note: __waitWd1() is still preserved in the code.
         */

        /*
         * There is a hardware bug in both watchdog timers.
         * If the STALL bit of the WDTTEST register is set,
         * the WTDLOAD register cannot be changed.
         * See page 68 of the Silicon Errata for more details.
         *
         * For that reason the STALL bit will be cleared first
         * and set again later.
         */
        HWREG_CLEAR_BITS( pReg[wd]->WD_TEST, TST_STALL );
        __waitWd1(wd);

        pReg[wd]->WD_LOAD = __wdSettings[wd].loadVal;
        __waitWd1(wd);

        /* Set the STALL bit again. */
        HWREG_SET_BITS( pReg[wd]->WD_TEST, TST_STALL );
        __waitWd1(wd);

    }
}


/*
 * An "unofficial" function (i.e. it should only be called from
 * startup.c and is therefore not publicly exposed in watchdog.h)
 * that checks which watchdog timer has triggered
 * an interrupt request, and runs its appropriate interrupt
 * handling routine as specified in __wdSettings[wd].isr =.
 */
__attribute__ ((interrupt))
void _wd_intHandler(void)
{
    /*
     * Poll both WDTMIS registers to find the watchdog
     * that triggered the interrupt
     * For more details about the WDTMIS register,
     * see page 785 of the Data Sheet.
     * 
     * WARNING: registers of a watchdog, not enabled in SYSCTL,
     * must not be accessed or a bus fault may occur!
     * For that reason, __enabled MUST be checked for each 'wd' first.
     * The if statement also checks if __intrIsr is equal to NULL.
     * If this is the case, this handler would not do anything anyway.
     */

    uint8_t wd;

    for ( wd=0; wd<BSP_NR_WATCHDOGS; ++wd )
    {
        if ( true == __wdSettings[wd].enabled &&
             NULL != __wdSettings[wd].intrIsr &&
             0 != HWREG_READ_BITS( pReg[wd]->WD_MIS, MIS_MASK ) )
        {		
            ( *(__wdSettings[wd].intrIsr) )();
           
            /*
             * The interrupt request source is deliberately
             * not cleared, this enables reset (if configured so)
             * on the second time out.
             */
        }  /* if */

    }  /* for wd */
}

/*
 * An "unofficial" function (i.e. it should only be called from
 * startup.c and is therefore not publicly exposed in watchdog.h)
 * that initializes both elements of __wdSettings.
 *
 * This function should only be called at the start of
 * an application.
 */
void _wd_initIntHandlers(void)
{
    uint8_t wd;

    for ( wd=0; wd<BSP_NR_WATCHDOGS; ++wd )
    {
        wd_unregisterIntHandler(wd);
        __wdSettings[wd].enabled = false;
        __wdSettings[wd].loadVal = 0xFFFFFFFF;
        __wdSettings[wd].exType = WDEX_IRQ;
    }
}
