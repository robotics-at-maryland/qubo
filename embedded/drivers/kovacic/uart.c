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
 * Implementation of the board's UART functionality.
 * All 8 UARTs are supported.
 *
 * For more info about the UART controller, see pp. 894 - 952 of:
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "FreeRTOSConfig.h"

#include "bsp.h"
#include "regutil.h"
#include "uart.h"
#include "sysctl.h"
#include "gpio.h"
#include "nvic.h"


/* Convenience bit masks for setting of clock divisors */
#define IBRD_MASK          ( 0x0000FFFF )
#define FBRD_MASK          ( 0x0000003F )

/* Bit masks for various settings at the Line Control Register (LCRH )*/
#define LCRH_FEN_MASK      ( 0x00000010 )
#define LCRH_WLEN_MASK     ( 0x00000060 )
#define LCRH_WLEN_SHIFT    ( 5 )
#define LCRH_PAR_MASK      ( 0x00000002 )
#define LCRH_EPS_MASK      ( 0x00000004 )
#define LCRH_SPS_MASK      ( 0x00000080 )
#define LCRH_STP2_MASK     ( 0x00000008 )

/* Bitmasks to configure triggering of interrupts: */
#define IFLS_RX_MASK       ( 0x00000038 )
#define IFLS_TX_MASK       ( 0x00000007 )
#define IFLS_RX_SHIFT      ( 3 )
#define IFLS_TX_SHIFT      ( 0 )
#define IFLS_RXFIFO_1_8_FULL      ( 0x0 )
#define IFLS_RXFIFO_1_4_FULL      ( 0x1 )
#define IFLS_RXFIFO_1_2_FULL      ( 0x2 )
#define IFLS_RXFIFO_3_4_FULL      ( 0x3 )
#define IFLS_RXFIFO_7_8_FULL      ( 0x4 )
#define IFLS_TXFIFO_7_8_EMPTY     ( 0x0 )
#define IFLS_TXFIFO_3_4_EMPTY     ( 0x1 )
#define IFLS_TXFIFO_1_2_EMPTY     ( 0x2 )
#define IFLS_TXFIFO_1_4_EMPTY     ( 0x3 )
#define IFLS_TXFIFO_1_8_EMPTY     ( 0x4 )

/* Bit masks for various settings at the Control Register: */
#define CTL_UART_ENABLE    ( 0x00000001 )
#define CTL_RX_ENABLE      ( 0x00000200 )
#define CTL_TX_ENABLE      ( 0x00000100 )

/* Bit masks for the Flag Register */
#define FR_TXFF            ( 0x00000020 )
#define FR_RXFE            ( 0x00000010 )
#define FR_TXFE            ( 0x00000080 )

/* Bit masks for the Clock Register CC */
#define CC_CS_MASK         ( 0x0000000F )
#define CC_PIOSC           ( 0x00000005 )
#define CC_SYSCLOCK        ( 0x00000000 )

/* Bit flags of interrupt handling registers (IM, ICR, RIS, MIS, etc.) */
#define INT_9BIT           ( 0x00001000 )
#define INT_OE             ( 0x00000400 )
#define INT_BE             ( 0x00000200 )
#define INT_PE             ( 0x00000100 )
#define INT_FE             ( 0x00000080 )
#define INT_RT             ( 0x00000040 )
#define INT_TX             ( 0x00000020 )
#define INT_RX             ( 0x00000010 )
#define INT_CTS            ( 0x00000002 )


/*
 * 32-bit Registers of individual UART controllers,
 * relative to the controller's base address:
 * See pages 905 - 906 of the Data Sheet.
 */
typedef struct _TM4C123G_UART_REGS
{
    uint32_t UART_DR;                  /* UART Data */
    uint32_t UART_RSR;                 /* UART Receive Status / Error Clear */
    const uint32_t Reserved1[4];       /* reserved */
    const uint32_t UART_FR;            /* UART Flag, read only */
    const uint32_t Reserved2;          /* reserved */
    uint32_t UART_ILP;                 /* UART IrDA Low-Power Register */
    uint32_t UART_IBRD;                /* UART Integer Baud-Rate Divisor */
    uint32_t UART_FBRD;                /* UART Fractional Baud-Rate Divisor */
    uint32_t UART_LCRH;                /* UART Line Control */
    uint32_t UART_CTL;                 /* UART Control */
    uint32_t UART_IFLS;                /* UART Interrupt FIFO Level Select */
    uint32_t UART_IM;                  /* UART Interrupt Mask */
    const uint32_t UART_RIS;           /* UART Raw Interrupt Status, read only */
    const uint32_t UART_MIS;           /* UART Masked Interrupt Status, read only */
    uint32_t UART_ICR;                 /* UART Interrupt Clear, write only */
    uint32_t UART_DMACTL;              /* UART DMA Control */
    const uint32_t Reserved3[22];      /* reserved */
    uint32_t UART_9BITADDR;            /* UART 9-Bit Self Address */
    uint32_t UART_9BITAMASK;           /* UART 9-Bit Self Address Mask */
    const uint32_t Reserved4[965];     /* reserved */
    const uint32_t UART_PP;            /* UART Peripheral Properties, read only */
    const uint32_t Reserved5;          /* reserved */
    uint32_t UART_CC;                  /* UART Clock Configuration */
    const uint32_t Reserved6;          /* reserved */
    const uint32_t UART_PeriphID4;     /* UART Peripheral Identification 4, read only */
    const uint32_t UART_PeriphID5;     /* UART Peripheral Identification 5, read only */
    const uint32_t UART_PeriphID6;     /* UART Peripheral Identification 6, read only */
    const uint32_t UART_PeriphID7;     /* UART Peripheral Identification 7, read only */
    const uint32_t UART_PeriphID0;     /* UART Peripheral Identification 0, read only */
    const uint32_t UART_PeriphID1;     /* UART Peripheral Identification 1, read only */
    const uint32_t UART_PeriphID2;     /* UART Peripheral Identification 2, read only */
    const uint32_t UART_PeriphID3;     /* UART Peripheral Identification 3, read only */
    const uint32_t UART_PCellID0;      /* UART PrimeCell Identification 0, read only */
    const uint32_t UART_PCellID1;      /* UART PrimeCell Identification 1, read only */
    const uint32_t UART_PCellID2;      /* UART PrimeCell Identification 2, read only */
    const uint32_t UART_PCellID3;      /* UART PrimeCell Identification 3, read only */
} TM4C123G_UART_REGS;

/* Shared UART register: */
#define UART_EC     UART_RS


/* ============================================================ */
#define GEN_CAST_ADDR(ADDR)     (volatile TM4C123G_UART_REGS* const) (ADDR),

static volatile TM4C123G_UART_REGS* const pReg[ BSP_NR_UARTS ]=
{
    BSP_UART_BASE_ADDRESSES( GEN_CAST_ADDR )
};

#undef GEN_CAST_ADDR
/* ============================================================ */

/* A vector of UARTs' IRQs: */
static uint8_t __uartIrqs[ BSP_NR_UARTS ] = BSP_UART_IRQS;


/**
 * Enables the selected UART.
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_enableUart(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        HWREG_SET_BITS( pReg[nr]->UART_CTL, CTL_UART_ENABLE );
    }
}


/**
 * Disables the selected UART.
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_disableUart(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        HWREG_CLEAR_BITS( pReg[nr]->UART_CTL, CTL_UART_ENABLE);
    }
}


/**
 * Waits until all waiting data are transmitted by the UART
 * and its transmit FIFO is empty.
 *
 * It is recommended to call this function before a UART
 * is reconfigured.
 *
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_flushTxFifo(uint8_t nr)
{
    uint32_t fen;

    if ( nr < BSP_NR_UARTS )
    {
        /* Store the current FEN flag */
        fen = HWREG_READ_BITS( pReg[nr]->UART_LCRH, LCRH_FEN_MASK );

        /* Flush the transmit FIFO by clearing the LCRH's FEN flag: */
        HWREG_CLEAR_BITS( pReg[nr]->UART_LCRH, LCRH_FEN_MASK );

        /* Wait until receive and transmit are complete */
        while
            ( (FR_TXFE | FR_RXFE) != HWREG_READ_BITS(pReg[nr]->UART_FR, FR_TXFE | FR_RXFE) );

        /* Restore the FEN flag*/
        HWREG_SET_BITS( pReg[nr]->UART_LCRH, fen );
    }
}


/*
 * As mentioned in the Data Sheet page 919, the Control Register
 * (CTL) should not be modified when the UART is enabled.
 * For that reason this inline function is introduced and will be
 * called by all other functions that handle the CTL register.
 *
 * The function checks for the current status of the UART (enabled
 * or disabled), then the UART is disabled, requested bits are set
 * or cleared, finally the status of the UART is restored.
 *
 * As the function is not "public" it does not perform a
 * sanity check and relies on calling functions that 'nr' is
 * within the allowed range (between 0 and 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param set - false: bitmask's bits are cleared to 0; true: bitmask's bit(s) are set to 1
 * @param bitmask - bitmask of 1-bits that will be set or cleared
 */
static inline void __setCtlBits(uint8_t nr, bool set, uint32_t bitmask)
{
    uint32_t enabled;

    /* Store UART's enable and FIFO enable status */
    enabled = HWREG_READ_BITS( pReg[nr]->UART_CTL, CTL_UART_ENABLE );

    /*
     * Disable the UART prior to any
     * modifications of the Control Register.
     */
    HWREG_CLEAR_BITS ( pReg[nr]->UART_CTL, CTL_UART_ENABLE );

    /* Flush the transmit FIFO: */
    uart_flushTxFifo(nr);


    /* Depending on 'set' ... */
    if ( true == set )
    {
        /* ... set bitmask's bits to 1 */
        HWREG_SET_BITS( pReg[nr]->UART_CTL, bitmask );
    }
    else
    {
        /* ... or clear bitmask's bits to 0 */
        HWREG_CLEAR_BITS( pReg[nr]->UART_CTL, bitmask );
    }

    /* Restore the original enable status */
    HWREG_SET_BITS( pReg[nr]->UART_CTL, enabled );
}

/**
 * Enables receive on the selected UART.
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_enableRx(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        __setCtlBits(nr, true, CTL_RX_ENABLE);
    }
}


/**
 * Disables receive on the selected UART.
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_disableRx(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        __setCtlBits(nr, false, CTL_RX_ENABLE);
    }
}


/**
 * Enables transmit on the selected UART.
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_enableTx(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        __setCtlBits(nr, true, CTL_TX_ENABLE);
    }
}


/**
 * Disables receive on the selected UART.
 * Nothing is done if 'nr' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_disableTx(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        __setCtlBits(nr, false, CTL_TX_ENABLE);
    }
}


/**
 * Enables the interrupt triggering by the specified UART
 * when a character is received.
 *
 * Nothing is done if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_enableRxIntr(uint8_t nr)
{
    /*
     * RX interrupt triggering is enabled by setting
     * the RXIM flag of the UARTIM register.
     * For more details, see page 926 of the Data Sheet.
     */

    if ( nr < BSP_NR_UARTS )
    {
        HWREG_SET_BITS( pReg[nr]->UART_IM, INT_RX );
    }
}


/**
 * Disables the interrupt triggering by the specified UART
 * when a character is received.
 *
 * Nothing is done if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_disableRxIntr(uint8_t nr)
{
    /*
     * RX interrupt triggering is enabled by clearing
     * the RXIM flag of the UARTIM register.
     * For more details, see page 926 of the Data Sheet.
     */

    if ( nr < BSP_NR_UARTS )
    {
        HWREG_CLEAR_BITS( pReg[nr]->UART_IM, INT_RX );
    }
}


/**
 * Clears Rx interrupt at the specified UART.
 * If the UART's receive FIFO is not empty yet, the
 * interrupt is not cleared.
 *
 * Nothing is done if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_clearRxIntr(uint8_t nr)
{
    /*
     * Rx interrupt is cleared by setting the RXIC bit of the
     * UART_ICR register. For more details, see
     * pp. 934 - 935 of the Data Sheet.
     */

    if ( nr < BSP_NR_UARTS && HWREG_READ_BITS( pReg[nr]->UART_FR, FR_RXFE ) )
    {
        /*
         * UART_ICR is a write only register, so usage of the
         * |= operator is not permitted. Anyway, zero-bits have
         * no effect on their corresponding interrupts so it
         * is perfectly OK to write the appropriate bit mask to
         * the register.
         */

        pReg[nr]->UART_ICR = INT_RX;
    }
}


/**
 * Enables character mode for the selected UART. In character mode,
 * internal FIFOs are disabled and become 1-byte-deep holding registers.
 * This is particularly useful when a UART is connected to a
 * text terminal.
 *
 * Nothing is done if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_characterMode(uint8_t nr)
{
    /*
     * Character mode (when FIFOs are disabled) is toggled by
     * clearing the FEN flag of the LCRH register.
     * See pp. 917 - 918 of the Data Sheet for more details.
     */

    if ( nr < BSP_NR_UARTS )
    {
        HWREG_CLEAR_BITS( pReg[nr]->UART_LCRH, LCRH_FEN_MASK );
    }
}


/**
 * Enables FIFO mode for the selected UART. In FIFO mode,
 * 16 transmit and receive buffers are enabled. Rx interrupts
 * are only triggered when a certain part (depending on 'level')
 * of receive FIFO is full.
 *
 * Nothing is done if 'nr' is invalid (greater than 7) or
 * 'level' has an invalid value.
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param level - UART receive interrupt FIFO level (a value of the rx_interrupt_fifo_level_t)
 */
void uart_fifoMode(uint8_t nr, rx_interrupt_fifo_level_t level)
{
    /*
     * FIFO mode is enabled by setting the FEN flag of
     * the LCRH register.
     * See pp. 917 - 918 of the Data Sheet for more details.
     *
     * Interrupt FIFO level select is additionally set by
     * appropriate bits of the IFLS register.
     * See pp. 923 - 924 of the Data Sheet for more details.
     */

    uint8_t rxiflsel = (uint8_t) -1;

    if ( nr < BSP_NR_UARTS )
    {

        switch (level)
        {
        case RXFIFO_1_8_FULL :
            rxiflsel = IFLS_RXFIFO_1_8_FULL;
            break;

        case RXFIFO_1_4_FULL :
            rxiflsel = IFLS_RXFIFO_1_4_FULL;
            break;

        case RXFIFO_1_2_FULL :
            rxiflsel = IFLS_RXFIFO_1_2_FULL;
            break;

        case RXFIFO_3_4_FULL :
        	rxiflsel = IFLS_RXFIFO_3_4_FULL;
            break;

        case RXFIFO_7_8_FULL :
            rxiflsel = IFLS_RXFIFO_7_8_FULL;
            break;
        } /* switch */

        /* Do nothing if 'level' is invalid */
        if ( rxiflsel < 0x5 )
        {
            /* Enable FIFO mode */
        	HWREG_SET_BITS( pReg[nr]->UART_LCRH, LCRH_FEN_MASK );

            /* And adjust appropriate bits of IFLS */
        	HWREG_SET_CLEAR_BITS( pReg[nr]->UART_IFLS,
        	                   (rxiflsel << IFLS_RX_SHIFT),
        	                   IFLS_RX_MASK );
        }

    } /* if nr < BSP_NR_UARTS */
}


/**
 * Unmasks the selected UART's IRQ requests on the NVIC.
 * Interrupts must be additionally enabled by setting
 * appropriate bits of the UARTIM register.
 *
 * Nothing is done if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_enableNvicIntr(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        nvic_enableInterrupt( __uartIrqs[nr] );
    }
}


/**
 * Masks the selected UART's IRQ requests on the NVIC.
 *
 * Nothing is done if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 */
void uart_disableNvicIntr(uint8_t nr)
{
    if ( nr < BSP_NR_UARTS )
    {
        nvic_disableInterrupt( __uartIrqs[nr] );
    }
}


/**
 * Sets priority of IRQ requests, triggered by the
 * selected UART.
 *
 * Nothing is done if eiter 'nr' or 'pri' is greater than 7.
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param pri - priority of interrupts triggered by the selected UART (between 0 and 7)
 */
void uart_setIntrPriority(uint8_t nr, uint8_t pri)
{
    if ( nr<BSP_NR_UARTS && pri<=MAX_PRIORITY )
    {
        nvic_setPriority( __uartIrqs[nr], pri );
    }
}


/**
 * Reads a character that was received by the specified UART.
 * The function may block until a character appears in the UART's receive buffer.
 * It is recommended that the function is called, when the caller is sure that a
 * character has actually been received, e.g. by notification via an interrupt.
 *
 * A zero is returned immediately if 'nr' is invalid (greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 *
 * @return character received at the UART
 */
char uart_readChar(uint8_t nr)
{
    /* Sanity check */
    if ( nr >= BSP_NR_UARTS )
    {
        return (char) 0;
    }

    /* Wait until the receiving FIFO is not empty */
    while ( HWREG_READ_BITS( pReg[nr]->UART_FR, FR_RXFE) );

    /*
     * UART DR is a 32-bit register and only the least significant byte
     * must be returned. Casting its address to char* effectively turns
     * the word into an array of (four) 8-bit characters. Now, dereferencing
     * the first character of this array affects only the desired character
     * itself, not the whole word.
     */

    return *( (char*) &(pReg[nr]->UART_DR) );
}


/*
 * Outputs a character to the specified UART. This short function is used by other functions,
 * that is why it is implemented as an inline function.
 *
 * As the function is "private", it relies on its caller functions, that 'nr'
 * is valid (between 0 and 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param ch - character to be sent to the UART
 */
static inline void __printCh(uint8_t uart, char ch)
{
    /*
     * Poll the Flag Register's TXFF bit until the Transmit FIFO is not full.
     * When the TXFF bit is set to 1, the controller's internal Transmit FIFO is full.
     * In this case, wait until some "waiting" characters have been transmitted and
     * the TXFF is set to 0, indicating the Transmit FIFO can accept additional characters.
     */
    while ( HWREG_READ_BITS(pReg[uart]->UART_FR, FR_TXFF) );

    /*
     * The Data Register is a 32-bit word, however only the least significant 8 bits
     * can be assigned the character to be sent, while other bits represent various flags
     * and should not be set to 0. For that reason, the following trick is introduced:
     *
     * Casting the Data Register's address to char* effectively turns the word into an array
     * of (four) 8-bit characters. Now, dereferencing the first character of this array affects
     * only the desired character itself, not the whole word.
     */
    *( (char*) &(pReg[uart]->UART_DR) ) = ch;
}


/**
 * Outputs a string to the specified UART.
 *
 * "<NULL>" is transmitted if 'str' is equal to NULL.
 *
 * Nothing is done if 'nr' is invalid (equal or greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param str - string to be sent to the UART, must be '\0' terminated.
 */
void uart_printStr(uint8_t nr, const char* str)
{
    /*
     * if NULL is passed, avoid possible problems with dereferencing of NULL
     * and print this string:
     */
    const char* null_str = "<NULL\r\n>";
    const char* cp;

    if ( nr < BSP_NR_UARTS )
    {
        /* handle possible NULL value of str: */
        cp = ( NULL==str ? null_str : (char*) str );

        /* Just print each character until a zero terminator is detected */
        for ( ; '\0'!=*cp; ++cp)
        {
            __printCh(nr, *cp);
        }
    }
}


/**
 * Outputs a character to the specified UART.
 *
 * Nothing is done if 'nr' is invalid (equal or greater than 7).
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param ch - character to be sent to the UART
 */
void uart_printCh(uint8_t nr, char ch)
{
    if ( nr < BSP_NR_UARTS )
    {
        /* just use the provided inline function: */
        __printCh(nr, ch);
    }
}


/**
 * Configures the UART to the desired baud rate, attempts to
 * connect it to desired pins on the selected GPIO ports, enables character
 * mode (no FIFOs), sets the 8 data bits, no parity, 1 stop protocol
 * (most commonly used). All interrupt sources are disabled (masked out).
 * Additionally the UART's IRQ requests are disabled on the NVIC.
 *
 * Note that this function does not enable the UART yet. After
 * the UART has been configured, its Rx and/or Tx functionality must
 * be enabled first, then the UART itself must be enabled.
 *
 * Nothing is done if any input value is out of range.
 *
 * Note: See the table 10.2 (Data Sheet, pp. 650 - 651) to
 * find out, which UARTs can be mapped to which GPIO pins.
 *
 * @param nr - number of the UART (between 0 and 7)
 * @param gp - GPIO port whose pins will be used (between 0 and 5)
 * @param pinRx - pin number the UART's Rx signal will be connected to (between 0 and 7)
 * @param pinTx - pin number the UART's Rx signal will be connected to (between 0 and 7)
 * @param pctl - a value that must be applied to pins' GPIOPCRTL registers in order to be connected to the UART
 * @param br - communication baud rate (a value of the enum baud_rate_t)
 * @param data_bits - number of data bits per frame (between 5 and 8)
 * @param parity - parity bit generation/detection (a value of the enum parity_t)
 * @param stop - 1 for 1 stop bit, anything else for 2 stop bits
 */
void uart_config(
        uint8_t nr,
        uint8_t gp,
        uint8_t pinRx,
        uint8_t pinTx,
        uint8_t pctl,
        baud_rate_t br,
        uint8_t data_bits,
        parity_t parity,
        uint8_t stop )
{
    uint32_t lcrh_r = 0x00000000;
    uint32_t divint = 0;
    uint32_t divrem = 0;


    if ( nr >= BSP_NR_UARTS ||
         gp >= BSP_NR_GPIO_PORTS  ||
         pinRx >= 8 ||
         pinTx >= 8 ||
         pctl > 15  ||
         ( data_bits<5 || data_bits>8 )
         )
    {
        return;
    }

    /*
     * Determine the clock divisors that suit the desired baud rate:
     *
     * To obtain the integer part of the divisor (divint), the UART clock
     * frequency must be divided by (16*desired_bauid_rate) and
     * rounded down:
     *
     *                   /  clock_speed   \
     *    divint = floor |----------------|
     *                   \ 16 * baud_rate /
     *
     * The fractional part of the divisor (divrem) is obtained
     * by multiplying of the fractional part of divint by 64 and
     * rounding the result to the nearest integer.
     *
     *                  /                                    \
     *                  | /    clock_speed            \      |
     *   divrem = round | | ---------------- - divint | * 64 |
     *                  | \  16 * baud_rate           /      |
     *                  \                                    /
     *
     * For more details, see page 897 of the Data Sheet.
     *
     *
     * The code below assumes, the clock source is the 16 MHz
     * precision internal oscillator (PIOSC) and all divint and
     * divrem values have been calculated based on that assumption.
     */

    switch (br)
    {
    case BR_9600:
        divint = 104;
        divrem = 11;
        break;

    case BR_19200:
        divint = 52;
        divrem = 5;
        break;

    case BR_38400:
        divint = 26;
        divrem = 3;
        break;

    case BR_57600:
        divint = 17;
        divrem = 23;
        break;

    case BR_115200:
        divint = 8;
        divrem = 44;
        break;

    default:
        /* Unsupported baud rate */
        return;
    }


    /*
     * Set the appropriate bits of the LCRH register
     * to configure the desired parity.
     *
     * More details at pages 917 - 918 of the Data Sheet
     */

    switch (parity)
    {
    case PAR_NONE:
        HWREG_CLEAR_BITS( lcrh_r, LCRH_PAR_MASK );
        break;

    case PAR_ODD:
        HWREG_SET_BITS( lcrh_r, LCRH_PAR_MASK );
        HWREG_CLEAR_BITS( lcrh_r, LCRH_EPS_MASK );
        break;

    case PAR_EVEN:
        HWREG_SET_BITS( lcrh_r, LCRH_PAR_MASK );
        HWREG_SET_BITS( lcrh_r, LCRH_EPS_MASK );
        break;

    case PAR_STICKY_0:
        HWREG_SET_BITS( lcrh_r, LCRH_PAR_MASK );
        HWREG_SET_BITS( lcrh_r, LCRH_EPS_MASK );
        HWREG_SET_BITS( lcrh_r, LCRH_SPS_MASK );
        break;

    case PAR_STICKY_1:
        HWREG_SET_BITS( lcrh_r, LCRH_PAR_MASK );
        HWREG_CLEAR_BITS( lcrh_r, LCRH_EPS_MASK );
        HWREG_SET_BITS( lcrh_r, LCRH_SPS_MASK );
        break;

    default:
        /* Unsupported parity */
        return;
    }

    /* Enable the UART at the System Control */
    sysctl_enableUart(nr);

    /* Enable the relevant GPIO port */
    sysctl_enableGpioPort(gp);

    /* Disable the UART during configuration: */
    uart_disableUart(nr);

    /* Select the PIOSC clock: */
    HWREG_SET_CLEAR_BITS( pReg[nr]->UART_CC, CC_PIOSC, CC_CS_MASK );

    /* Set the baud rate: */
    HWREG_SET_CLEAR_BITS( pReg[nr]->UART_IBRD, divint, IBRD_MASK );
    HWREG_SET_CLEAR_BITS( pReg[nr]->UART_FBRD, divrem, FBRD_MASK);

    /* Disable FIFOs: */
    HWREG_CLEAR_BITS( lcrh_r, LCRH_FEN_MASK );

    /*
     * Mask out all UART's interrupt sources by clearing
     * all corresponding bits of UARTIM to 0:
     */
    HWREG_CLEAR_BITS( pReg[nr]->UART_IM,
        ( INT_9BIT |
          INT_OE   |
          INT_BE   |
          INT_PE   |
          INT_FE   |
          INT_RT   |
          INT_TX   |
          INT_RX   |
          INT_CTS ) );

    /* Set the number of data bits per frame */
    HWREG_SET_CLEAR_BITS( lcrh_r, (data_bits-5) << LCRH_WLEN_SHIFT, LCRH_WLEN_MASK);

    /* Set the desired number of stop bits */
    if ( 1 == stop )
    {
        HWREG_CLEAR_BITS( lcrh_r, LCRH_STP2_MASK );
    }
    else
    {
        HWREG_SET_BITS( lcrh_r, LCRH_STP2_MASK );
    }

    /* Finally update non-reserved bits of the UART_LCRH register */
    HWREG_SET_CLEAR_BITS( pReg[nr]->UART_LCRH, lcrh_r, 0x000000FF );


    /* Appropriately configure GPIO to map Rx and Tx pins to UART: */
    gpio_setAltFunction(gp, pinRx, pctl);
    gpio_setAltFunction(gp, pinTx, pctl);

    gpio_disableAnalog(gp, pinRx);
    gpio_disableAnalog(gp, pinTx);

    gpio_enableDigital(gp, pinRx);
    gpio_enableDigital(gp, pinTx);

    /* Mask UART's IRQs in NVIC: */
    uart_setIntrPriority(nr, APP_DEF_UART_IRQ_PRIORITY);
    uart_disableNvicIntr(nr);
}
