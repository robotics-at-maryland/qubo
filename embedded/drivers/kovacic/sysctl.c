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
 * Implementation of the board's System Controller
 * (SysCtl) functionality.
 *
 * For more info about the system controller, see pp. 212 - 484 of:
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
 * 32-bit Registers of the system controller,
 * relative to the controller's base address:
 * See pages 231 - 237 of the Data Sheet.
 * The struct does not include legacy registers.
 */
typedef struct _TM4C123G_SYSCTL_REGS
{
    const uint32_t SYSCTL_DID0;      /* Device Identification 0, read only */
    const uint32_t SYSCTL_DID1;      /* Device Identification 1, read only */
    const uint32_t Reserved1[10];    /* reserved */
    uint32_t SYSCTL_PBORCTL;         /* Brown-Out Reset Control*/
    const uint32_t Reserved2[7];     /* reserved */
    const uint32_t SYSCTL_RIS;       /* Raw Interrupt Status, read only */
    uint32_t SYSCTL_IMC;             /* Interrupt Mask Control */
    uint32_t SYSCTL_MISC;            /* Masked Interrupt Status and Clear */
    uint32_t SYSCTL_RESC;            /* Reset Cause */
    uint32_t SYSCTL_RCC;             /* Run-Mode Clock Configuration */
    const uint32_t Reserved3[2];     /* reserved */
    uint32_t SYSCTL_GPIOHBCTL;       /* GPIO High-Performance Bus Control */
    uint32_t SYSCTL_RCC2;            /* Run-Mode Clock Configuration 2 */
    const uint32_t Reserved4[2];     /* reserved */
    uint32_t SYSCTL_MOSCCTL;         /* Main Oscillator Control */
    const uint32_t Reserved5[49];    /* reserved */
    uint32_t SYSCTL_DSLPCLKCFG;      /* Deep Sleep Clock Configuration */
    const uint32_t Reserved6;        /* reserved */
    const uint32_t SYSCTL_SYSPROP;   /* System Properties, read only */
    uint32_t SYSCTL_PIOSCCAL;        /* Precision Internal Oscillator Calibration */
    const uint32_t SYSCTL_PIOSCSTAT; /* Precision Internal Oscillator Statistics, read only */
    const uint32_t Reserved7[2];     /* reserved */
    const uint32_t SYSCTL_PLLFREQ0;  /* PLL Frequency 0, read only */
    const uint32_t SYSCTL_PLLFREQ1;  /* PLL Frequency 1, read only */
    const uint32_t SYSCTL_PLLSTAT;   /* PLL Status, read only */
    const uint32_t Reserved8[7];     /* reserved */
    uint32_t SYSCTL_SLPPWRCFG;       /* Sleep Power Configuration */
    uint32_t SYSCTL_DSLPPWRCFG;      /* Deep Sleep Power Configuration */
    const uint32_t Reserved9[9];     /* reserved */
    uint32_t SYSCTL_LDOSPCTL;        /* LDO Sleep Power Control */
    const uint32_t SYSCTL_LDOSPCAL;  /* LDO Sleep Power Calibration, read only */
    uint32_t SYSCTL_LDODPCTL;        /* LDO Deep-Sleep Power Control */
    const uint32_t SYSCTL_LDODPCAL;  /* LDO Deep-Sleep Power Calibration, read only */
    const uint32_t Reserved10[2];    /* reserved */
    const uint32_t SYSCTL_SDPMST;    /* Sleep / Deep-Sleep Power Mode Status, read only */
    const uint32_t Reseved11[76];    /* reserved */
    const uint32_t SYSCTL_PPWD;      /* Watchdog Timer Peripheral Present, read only */
    const uint32_t SYSCTL_PPTIMER;   /* 16/32-Bit General-Purpose Timer Peripheral Present, read only */
    const uint32_t SYSCTL_PPGPIO;    /* General Purpose Input/Output Peripheral Present, read only */
    const uint32_t SYSCTL_PPDMA;     /* Micro Direct Memory Access Peripheral Present, read only */
    const uint32_t Reserved12;       /* reserved */
    const uint32_t SYSCTL_PPHIB;     /* Hibernation Peripheral Present, read only */
    const uint32_t SYSCTL_PPUART;    /* Universal Asynchronous Receiver/Transmitter Peripheral Present, read only */
    const uint32_t SYSCTL_PPSI;      /* Synchronous Serial Interface Peripheral Present, read only */
    const uint32_t SYSCTL_PPI2C;     /* Inter-Integrated Circuit Peripheral Present, read only */
    const uint32_t Reserved13;       /* reserved */
    const uint32_t SYSCTL_PPUSB;     /* Universal Serial Bus Peripheral Present, read only */
    const uint32_t Reserved14[2];    /* reserved */
    const uint32_t SYSCTL_PPCAN;     /* Controller Area Network Perpheral Present, read only */
    const uint32_t SYSCTL_PPADC;     /* Analog-to-Digital Converter Peripheral Present, read only */
    const uint32_t SYSCTL_PPACMP;    /* Analog Comparator Peripheral Present, read only */
    const uint32_t SYSCTL_PPPWM;     /* Pulse Width Modulator Peripheral Present, read only */
    const uint32_t SYSCTL_PPQEI;     /* Quadrature Encoder Interface Periphral Present, read only */
    const uint32_t Reserved15[4];    /* reserved */
    const uint32_t SYSCTL_PPEEPROM;  /* EEPROM Peripheral Present, read only */
    const uint32_t SYSCTL_PPWTIMER;  /* 32/64-Bit Wide General-Purpose Timer Peripheral Present, read only */
    const uint32_t Reserved16[104];  /* reserved */
    uint32_t SYSCTL_SRWD;            /* Watchdog Timer Software Reset */
    uint32_t SYSCTL_SRTIMER;         /* 16/32-Bit General-Purpose Timer Software Reset */
    uint32_t SYSCTL_SRGPIO;          /* General-Purpose Input/Output Software Reset */
    uint32_t SYSCTL_SRDMA;           /* Micro Direct Memory Access Software Reset */
    const uint32_t Reserved17;       /* reserved */
    uint32_t SYSCTL_SRHIB;           /* Hibernation Software Reset */
    uint32_t SYSCTL_SRUART;          /* Universal Asynchronous Receiver/Transmitter Software Reset */
    uint32_t SYSCTL_SRSSI;           /* Synchronous Serial Interface Software Reset */
    uint32_t SYSCTL_SRI2C;           /* Inter-Integrated Circuit Software Reset */
    const uint32_t Reserved18;       /* reserved */
    uint32_t SYSCTL_SRUSB;           /* Universal Serial Bus Software Reset */
    const uint32_t Reserved19[2];    /* reserved */
    uint32_t SYSCTL_SRCAN;           /* Controller Area Network Software Reset */
    uint32_t SYSCTL_SRADC;           /* Analog-to-Digital Converter Software Reset */
    uint32_t SYSCTL_SRACMP;          /* Analog Comparator Software Reset */
    uint32_t SYSCTL_SRPWM;           /* Pulse Width Modulator Software Reset */
    uint32_t SYSCTL_SRQEI;           /* Quadrature Encoder Interface Software Reset */
    const uint32_t Reserved20[4];    /* reserved */
    uint32_t SYSCTL_SREEPROM;        /* EEPROM Software Reset */
    uint32_t SYSCTL_SRWTIMER;        /* 32/64-Bit Wide General-Purpose Timer Software Reset*/
    const uint32_t Reserved21[40];   /* reserved */
    uint32_t SYSCTL_RCGCWD;          /* Watchdog Timer Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCTIMER;       /* 32/64-Bit General-Purpose Timer Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCGPIO;        /* General-Purpose Input/Output Timer Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCDMA;         /* Micro Direct Memory Access Timer Run Mode Clock Gating Control */
    const uint32_t Reserved22;       /* reserved */
    uint32_t SYSCTL_RCGCHIB;         /* Hibernation Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCUART;        /* Universal Asynchronous Receiver/Transmitter Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCSSI;         /* Synchronous Serial Interface Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCI2C;         /* Inter-Integrated Circuit Run Mode Clock Gating Control */
    const uint32_t Reserved23;       /* reserved */
    uint32_t SYSCTL_RCGCUSB;         /* Universal Serial Bus Run Mode Clock Gating Control */
    const uint32_t Reserved24[2];    /* reserved */
    uint32_t SYSCTL_RCGCCAN;         /* Controller Area Network Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCADC;         /* Analog-to-Digital Converter Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCACMP;        /* Analog Comparator Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCPWM;         /* Pulse Width Modulator Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCQEI;         /* Quadrature Encoder Interface Run Mode Clock Gating Control */
    const uint32_t Reserved25[4];    /* reserved */
    uint32_t SYSCTL_RCGCEEPROM;      /* EEPROM Run Mode Clock Gating Control */
    uint32_t SYSCTL_RCGCWTIMER;      /* 32/64-Bit Wide General-Purpose Timer Run Mode Clock Gating Control */
    const uint32_t Reserved26[40];   /* reserved */
    uint32_t SYSCTL_SCGCWD;          /* Watchdog Timer Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCTIMER;       /* 32/64-Bit General-Purpose Timer Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCGPIO;        /* General-Purpose Input/Output Timer Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCDMA;         /* Micro Direct Memory Access Timer Sleep Mode Clock Gating Control */
    const uint32_t Reserved27;       /* reserved */
    uint32_t SYSCTL_SCGCHIB;         /* Hibernation Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCUART;        /* Universal Asynchronous Receiver/Transmitter Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCSSI;         /* Synchronous Serial Interface Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCI2C;         /* Inter-Integrated Circuit Sleep Mode Clock Gating Control */
    const uint32_t Reserved28;       /* reserved */
    uint32_t SYSCTL_SCGCUSB;         /* Universal Serial Bus Sleep Mode Clock Gating Control */
    const uint32_t Reserved29[2];    /* reserved */
    uint32_t SYSCTL_SCGCCAN;         /* Controller Area Network Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCADC;         /* Analog-to-Digital Converter Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCACMP;        /* Analog Comparator Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCPWM;         /* Pulse Width Modulator Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCQEI;         /* Quadrature Encoder Interface Sleep Mode Clock Gating Control */
    const uint32_t Reserved30[4];    /* reserved */
    uint32_t SYSCTL_SCGCEEPROM;      /* EEPROM Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_SCGCWTIMER;      /* 32/64-Bit Wide General-Purpose Timer Sleep Mode Clock Gating Control */
    const uint32_t Reserved31[40];   /* reserved */
    uint32_t SYSCTL_DCGCWD;          /* Watchdog Timer Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCTIMER;       /* 32/64-Bit General-Purpose Timer Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCGPIO;        /* General-Purpose Input/Output Timer Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCDMA;         /* Micro Direct Memory Access Timer Deep-Sleep Mode Clock Gating Control */
    const uint32_t Reserved32;       /* reserved */
    uint32_t SYSCTL_DCGCHIB;         /* Hibernation Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCUART;        /* Universal Asynchronous Receiver/Transmitter Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCSSI;         /* Synchronous Serial Interface Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCI2C;         /* Inter-Integrated Circuit Deep-Sleep Mode Clock Gating Control */
    const uint32_t Reserved33;       /* reserved */
    uint32_t SYSCTL_DCGCUSB;         /* Universal Serial Bus Deep-Sleep Mode Clock Gating Control */
    const uint32_t Reserved34[2];    /* reserved */
    uint32_t SYSCTL_DCGCCAN;         /* Controller Area Network Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCADC;         /* Analog-to-Digital Converter Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCACMP;        /* Analog Comparator Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCPWM;         /* Pulse Width Modulator Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCQEI;         /* Quadrature Encoder Interface Deep-Sleep Mode Clock Gating Control */
    const uint32_t Reserved35[4];    /* reserved */
    uint32_t SYSCTL_DCGCEEPROM;      /* EEPROM Deep-Sleep Mode Clock Gating Control */
    uint32_t SYSCTL_DCGCWTIMER;      /* 32/64-Bit Wide General-Purpose Timer Deep-Sleep Mode Clock Gating Control */
    const uint32_t Reserved36[104];  /* reserved */
    const uint32_t SYSCTL_PRWD;      /* Watchdog Timer Peripheral Ready */
    const uint32_t SYSCTL_PRTIMER;   /* 32/64-Bit General-Purpose Timer Peripheral Ready */
    const uint32_t SYSCTL_PRGPIO;    /* General-Purpose Input/Output Timer Peripheral Ready */
    const uint32_t SYSCTL_PRDMA;     /* Micro Direct Memory Access Timer Peripheral Ready */
    const uint32_t Reserved37;       /* reserved */
    const uint32_t SYSCTL_PRHIB;     /* Hibernation Peripheral Ready */
    const uint32_t SYSCTL_PRUART;    /* Universal Asynchronous Receiver/Transmitter Peripheral Ready */
    const uint32_t SYSCTL_PRSSI;     /* Synchronous Serial Interface Peripheral Ready */
    const uint32_t SYSCTL_PRI2C;     /* Inter-Integrated Circuit Peripheral Ready */
    const uint32_t Reserved38;       /* reserved */
    const uint32_t SYSCTL_PRUSB;     /* Universal Serial Bus Peripheral Ready */
    const uint32_t Reserved39[2];    /* reserved */
    const uint32_t SYSCTL_PRCAN;     /* Controller Area Network Peripheral Ready */
    const uint32_t SYSCTL_PRADC;     /* Analog-to-Digital Converter Peripheral Ready */
    const uint32_t SYSCTL_PRACMP;    /* Analog Comparator Peripheral Ready */
    const uint32_t SYSCTL_PRPWM;     /* Pulse Width Modulator Peripheral Ready */
    const uint32_t SYSCTL_PRQEI;     /* Quadrature Encoder Interface Peripheral Ready */
    const uint32_t Reserved40[4];    /* reserved */
    const uint32_t SYSCTL_PREEPROM;  /* EEPROM Peripheral Ready */
    const uint32_t SYSCTL_PRWTIMER;  /* 32/64-Bit Wide General-Purpose Timer Peripheral Ready */
} TM4C123G_SYSCTL_REGS;


static volatile TM4C123G_SYSCTL_REGS* const pReg =
   (volatile TM4C123G_SYSCTL_REGS* const) ( BSP_SYSCTL_BASE_ADDRESS );



/* A convenience 6-bit mask, handy for modifying the GPIOHBCTL register: */
#define GPIO_HBCTL_MASK      ( 0x0000003F )

/* Convenience bit masks for setting the RCC_XTAL register: */
#define RCC_XTAL_MASK        ( 0x000007C0 )
#define RCC_XTAL_OFFSET      ( 6 )


/* Convenience bit masks for setting of various RCC2* registers */
#define RCC2_USERCC2_MASK    ( 0x80000000 )
#define RCC2_BYPASS2_MASK    ( 0x00000800 )
#define RCC2_OSCSRC2_MASK    ( 0x00000070 )
#define RCC2_PWRDN2_MASK     ( 0x00002000 )
#define RCC2_DIV400_MASK     ( 0x40000000 )
#define RCC2_SYSDIV2_MASK    ( 0x1F800000 )
#define RCC2_SYSDIV2LSB_MASK ( 0x00400000 )
#define RCC2_SYSDIV2_SHIFT           ( 23 )
#define RCC2_SYSDIV2LSB_SHIFT        ( 22 )

/* Convenience bit mask for reading the appropriate bits of the PLLLRIS register */
#define RIS_PLLLRIS_MASK     ( 0x00000040 )

/* Microcontroller revision not determined yet: */
#define MCU_REV_NOT_KNOWN_YET      ( -128 )

/* Unknown microcontroller revision: */
#define MCU_REV_UNKNOWN              ( -1 )


/**
 *  Microcontroller's part revision number.
 *  This variable is automatically initialized
 *  during startup and may be declared as an external
 *  variable where needed.
 */
int8_t sysctl_mcu_revision = MCU_REV_NOT_KNOWN_YET;


/**
 * Microcontroller's part revision number is an important
 * piece of information, essential for implementation of
 * workarounds of known hardware bugs, described in the
 * Tiva(TM) C Series TM4C123x Microcontrollers Silicon Revisions 6 and 7,
 *     Silicon Errata
 * (http://www.ti.com/lit/er/spmz849c/spmz849c.pdf).
 *
 * @return microcontroller's part revision number (between 1 and 7 or a negative value if unknown)
 */
int8_t sysctl_mcuRevision(void)
{
    /*
     * The revision number can be determined from the DID0
     * register. See page 238 of the Data Sheet
     * for more details.
     */

    uint8_t minor;
    uint8_t major;

    if ( MCU_REV_NOT_KNOWN_YET == sysctl_mcu_revision )
    {
        minor = (uint8_t) (pReg->SYSCTL_DID0 & 0x000000FF);
        major = (uint8_t) ((pReg->SYSCTL_DID0 & 0x0000FF00) >> 8);

        sysctl_mcu_revision = MCU_REV_UNKNOWN;

        if ( 0==major && minor<=3 )
        {
            sysctl_mcu_revision = minor +1;
        }

        if ( 1==major && minor<=2 )
        {
            sysctl_mcu_revision = minor + 5;
        }
    }

    return sysctl_mcu_revision;
}


/**
 * Configures the system clock.
 *
 * The main oscillator (MOSC) is selected and Phase-Locked Loop (PLL)
 * is enabled.
 *
 * The system clock frequency is equal to 400 MHz, divided by 'div'.
 * For instance:
 * - for system clock frequency of 80 MHz, enter 'div' = 400 Mhz / 80 MHz = 5
 * - for system clock frequency of 50 MHz, enter 'div' = 400 Mhz / 50 MHz = 80
 * - for system clock frequency of 16 MHz, enter 'div' = 400 Mhz / 16 MHz = 25
 * etc.
 *
 * 'div' can be any integer value betwwen 5 and 128 incl., except 7.
 *
 * Typically a definition from pll_freq_divisors.h will be applied
 * to map a supported frequency to its corresponding divisor.
 *
 * If 'div' is invalid, system clock will be set to 16 MHz.
 *
 * @param div - integer divisor to divide 400 MHz (between 5 and 128; except 7)
 *
 * @return 1 if 'div' is valid, 0 if not (and frequency was set to 16 MHz)
 *
 * @see pll_freq_divisor.h
 */
uint8_t sysctl_configSysClock(uint8_t div)
{
    /*
     * For more details about configuration of the system clock,
     * see page 231 of the Data Sheet.
     */

    uint8_t sysdiv2;
    uint8_t sysdiv2lsb;
    uint8_t retVal = 1;

    /*
     * Determine values of RCC2's regions sysdiv2 and sysdiv2lsb.
     * See table 5-6 at page 224 of the Data Sheet for more details.
     *
     * 'sysdiv2' and 'sysdiv2lsb' can be evidently calculated as:
     *
     *       sysdiv2 = ( div - 1 ) / 2      (apply integer division)
     *    sysdiv2lsb = ( div - 1 ) % 2
     */
    if ( div<5 || div>128 || 7==div )
    {
        /*
         * Invalid values, set frequency of 16 Mhz,
         * div = 400 MHz / 16 MHz = 25
         */
        sysdiv2 = 12;
        sysdiv2lsb = 0;
        retVal = 0;
    }
    else
    {
        sysdiv2 = (div-1) / 2;
        sysdiv2lsb = (div-1) % 2;
        retVal = 1;
    }

    /* override RCC register fields: */
    HWREG_SET_BITS( pReg->SYSCTL_RCC2, RCC2_USERCC2_MASK );

    /* the system clock is currently derived from OSC: */
    HWREG_SET_BITS( pReg->SYSCTL_RCC2, RCC2_BYPASS2_MASK );

    /* reset the XTAL bits and select the 16 MHz crystal: */
    HWREG_SET_CLEAR_BITS( pReg->SYSCTL_RCC, (0x15 << RCC_XTAL_OFFSET), RCC_XTAL_MASK );

    /* clear all OSCSRC2 bits and thus select the main oscillator: */
    HWREG_CLEAR_BITS( pReg->SYSCTL_RCC2, RCC2_OSCSRC2_MASK );

    /* additionally enable the PLL by clearing the PWRDN2 bit: */
    HWREG_CLEAR_BITS( pReg->SYSCTL_RCC2, RCC2_PWRDN2_MASK );

    /* SYSDIV will divide 400 MHz instead of 200 MHz: */
    HWREG_SET_BITS( pReg->SYSCTL_RCC2, RCC2_DIV400_MASK );

    /* clear all SYSDIV2 and SYSDIV2LSB bits and set the desired frequency: */
    HWREG_SET_CLEAR_BITS(
            pReg->SYSCTL_RCC2,
            ( (sysdiv2 << RCC2_SYSDIV2_SHIFT) | (sysdiv2lsb << RCC2_SYSDIV2LSB_SHIFT) ),
            (RCC2_SYSDIV2_MASK | RCC2_SYSDIV2LSB_MASK) );

    /* wait until USB PLL locks: */
    while ( 0 == HWREG_READ_BITS(pReg->SYSCTL_RIS, RIS_PLLLRIS_MASK) );

    /* switch system clock to PLL: */
    HWREG_CLEAR_BITS( pReg->SYSCTL_RCC2, RCC2_BYPASS2_MASK);

    return (retVal);
}


/**
 * Enables the selected GPIO port in Run Mode.
 *
 * When enabled, the port is provided a clock and
 * access to its registers is allowed.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - GPIO port to be enabled (between 0 and 5)
 */
void sysctl_enableGpioPort(uint8_t port)
{
    /*
     * See pages 340 - 341 of the Data Sheet.
     */

    uint32_t delay;

    if ( port < BSP_NR_GPIO_PORTS )
    {
        if ( !HWREG_READ_SINGLE_BIT( pReg->SYSCTL_RCGCGPIO, port ) )
        {
            HWREG_SET_SINGLE_BIT( pReg->SYSCTL_RCGCGPIO, port );
        }

        /* allow a few CPU cycles until the port is finally configured: */
        for ( delay=0; delay<10; ++delay );
    }
}


/**
 * Disables the selected GPIO port in Run Mode.
 *
 * When disabled, the port is disconnected from the clock and
 * access to its registers is not allowed. Any attempt of
 * accessing a disabled port may result in a bus fault.
 *
 * Nothing is done if 'port' is greater than 5.
 *
 * @param port - GPIO port to be disabled (between 0 and 5)
 */
void sysctl_disableGpioPort(uint8_t port)
{
    /*
     * See pages 340 - 341 of the Data Sheet.
     */

    uint32_t delay;

    if ( port < BSP_NR_GPIO_PORTS )
    {
        if ( HWREG_READ_SINGLE_BIT( pReg->SYSCTL_RCGCGPIO, port ) )
        {
            HWREG_CLEAR_SINGLE_BIT( pReg->SYSCTL_RCGCGPIO, port );
        }
    }

    /* allow a few CPU cycles until the port is finally configured: */
    for ( delay=0; delay<10; ++delay );
}


/**
 * An "unofficial" function (not exposed in a header) that
 * enables access to all GPIO ports across the Advanced High-Performance
 * Bus (AHB) and AHB memory aperture.
 *
 * The function should only be called immediately when an application
 * starts if AHB is desired.
 */
void _sysctl_enableGpioAhb(void)
{
    /*
     * See pp. 258 - 259 of the Data Sheet
     */
    HWREG_SET_BITS( pReg->SYSCTL_GPIOHBCTL, GPIO_HBCTL_MASK );
}


/**
 * Enables the selected UART in Run Mode.
 *
 * When enabled, the UART is provided a clock and
 * access to its registers is allowed.
 *
 * Nothing is done if 'uartNr' is greater than 7.
 *
 * @param uartNr - number of the UART to be enabled (between 0 and 7)
 */
void sysctl_enableUart(uint8_t uartNr)
{
    /*
     * See pp. 344 - 345 of the Data Sheet
     */

    if ( uartNr < BSP_NR_UARTS )
    {
        if ( !HWREG_READ_SINGLE_BIT( pReg->SYSCTL_RCGCUART, uartNr ) )
        {
            HWREG_SET_SINGLE_BIT( pReg->SYSCTL_RCGCUART, uartNr);
        }
    }
}


/**
 * Disables the selected UART in Run Mode.
 *
 * When disabled, the UART is disconnected from the clock and
 * access to its registers is not allowed. Any attempt of
 * accessing a disabled UART may result in a bus fault.
 *
 * Nothing is done if 'uartNr' is greater than 7.
 *
 * @param uartNr - number of the UART to be disabled (between 0 and 7)
 */
void sysctl_disableUart(uint8_t uartNr)
{
    /*
     * See pp. 344 - 345 of the Data Sheet
     */

    if ( uartNr < BSP_NR_UARTS )
    {
        if ( HWREG_READ_SINGLE_BIT( pReg->SYSCTL_RCGCUART, uartNr ) )
        {
            HWREG_CLEAR_SINGLE_BIT( pReg->SYSCTL_RCGCUART, uartNr);
        }
    }
}


/**
 * Enables the selected Watchdog Timer in Run Mode.
 *
 * When enabled, the watchdog timer is provided a clock
 * and accesses to module registers are allowed.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - number of the watchdog timer to be enabled (between 0 and 1)
 */
void sysctl_enableWatchdog(uint8_t wd)
{
    /*
     * See page 337 of the Data Sheet
     */

    if ( wd < BSP_NR_WATCHDOGS )
    {
        if ( !HWREG_READ_SINGLE_BIT( pReg->SYSCTL_RCGCWD, wd) )
        {
            HWREG_SET_SINGLE_BIT( pReg->SYSCTL_RCGCWD, wd );
        }
    }
}


/**
 * Disables the selected Watchdog Timer in Run Mode.
 *
 * When disabled, the watchdog timer is disconnected from
 * the clock and access to its registers is not allowed.
 * Any attempt of accessing a disabled watchdog timer may
 * result in a bus fault.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - number of the watchdog timer to be enabled (between 0 and 1)
 */
void sysctl_disableWatchdog(uint8_t wd)
{
    /*
     * See page 337 of the Data Sheet
     */

    if ( wd < BSP_NR_WATCHDOGS )
    {
        if ( HWREG_READ_SINGLE_BIT( pReg->SYSCTL_RCGCWD, wd) )
        {
            HWREG_CLEAR_SINGLE_BIT( pReg->SYSCTL_RCGCWD, wd );
        }
    }
}


/**
 * Resets the selected watchdog module.
 *
 * Nothing is done if 'wd' is greater than 1.
 *
 * @param wd - number of the watchdog timer to be reset (between 0 and 1)
 */
void sysctl_resetWatchdog(uint8_t wd)
{
    /*
     * As described on pp. 310 - 311 of the Data Sheet,
     * reset of a watchdog module is completed in two steps.
     * - Corresponding bit of the SWRD register is set to 1.
     * - Corresponding bit of the SRWD register is cleared to 0.
     */

    if ( wd < BSP_NR_WATCHDOGS )
    {
        HWREG_SET_SINGLE_BIT( pReg->SYSCTL_SRWD, wd );
        HWREG_CLEAR_SINGLE_BIT( pReg->SYSCTL_SRWD, wd );
    }
}
