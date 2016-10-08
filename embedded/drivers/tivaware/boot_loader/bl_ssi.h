//*****************************************************************************
//
// bl_ssi.h - Definitions for the SSI transport functions.
//
// Copyright (c) 2006-2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
// 
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
// 
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************

#ifndef __BL_SSI_H__
#define __BL_SSI_H__

//*****************************************************************************
//
// This section maps the defines to default for SSI Boot Loader for legacy
// projects
//
//*****************************************************************************
#ifndef SSI_CLOCK_ENABLE
#define SSI_CLOCK_ENABLE        SYSCTL_RCGCSSI_R0
#endif

#ifndef SSIx_BASE
#define SSIx_BASE               SSI0_BASE
#endif

#ifndef SSI_CLKPIN_CLOCK_ENABLE
#define SSI_CLKPIN_CLOCK_ENABLE SYSCTL_RCGCGPIO_R0
#endif

#ifndef SSI_CLKPIN_BASE
#define SSI_CLKPIN_BASE         GPIO_PORTA_BASE
#endif

#ifndef SSI_CLKPIN_PCTL
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
#define SSI_CLKPIN_PCTL         0xF
#else
#define SSI_CLKPIN_PCTL         0x2
#endif
#endif

#ifndef SSI_CLKPIN_POS
#define SSI_CLKPIN_POS          2
#endif

#ifndef SSI_FSSPIN_CLOCK_ENABLE
#define SSI_FSSPIN_CLOCK_ENABLE SYSCTL_RCGCGPIO_R0
#endif

#ifndef SSI_FSSPIN_BASE
#define SSI_FSSPIN_BASE         GPIO_PORTA_BASE
#endif

#ifndef SSI_FSSPIN_PCTL
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
#define SSI_FSSPIN_PCTL         0xF
#else
#define SSI_FSSPIN_PCTL         0x2
#endif
#endif

#ifndef SSI_FSSPIN_POS
#define SSI_FSSPIN_POS          3
#endif

#ifndef SSI_MISOPIN_CLOCK_ENABLE
#define SSI_MISOPIN_CLOCK_ENABLE    SYSCTL_RCGCGPIO_R0
#endif

#ifndef SSI_MISOPIN_BASE
#define SSI_MISOPIN_BASE        GPIO_PORTA_BASE
#endif

#ifndef SSI_MISOPIN_PCTL
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
#define SSI_MISOPIN_PCTL        0xF
#else
#define SSI_MISOPIN_PCTL        0x2
#endif
#endif

#ifndef SSI_MISOPIN_POS
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
#define SSI_MISOPIN_POS         4
#else
#define SSI_MISOPIN_POS         5
#endif
#endif

#ifndef SSI_MOSIPIN_CLOCK_ENABLE
#define SSI_MOSIPIN_CLOCK_ENABLE SYSCTL_RCGCGPIO_R0
#endif

#ifndef SSI_MOSIPIN_BASE
#define SSI_MOSIPIN_BASE        GPIO_PORTA_BASE
#endif

#ifndef SSI_MOSIPIN_PCTL
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
#define SSI_MOSIPIN_PCTL        0xF
#else
#define SSI_MOSIPIN_PCTL        0x2
#endif
#endif

#ifndef SSI_MOSIPIN_POS
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
#define SSI_MOSIPIN_POS         5
#else
#define SSI_MOSIPIN_POS         4
#endif
#endif

//*****************************************************************************
//
// This is the number of bits per transfer for SSI.  This is a constant and
// cannot be changed without corresponding code changes.
//
//*****************************************************************************
#define DATA_BITS_SSI           8

//*****************************************************************************
//
// This defines the SSI chip select pin that is being used by the boot loader.
//
//*****************************************************************************
#define SSI_CS                  (1 << SSI_FSSPIN_POS)
#define SSI_CS_PCTL             (SSI_FSSPIN_PCTL << (4 * SSI_FSSPIN_POS))

//*****************************************************************************
//
// This defines the SSI clock pin that is being used by the boot loader.
//
//*****************************************************************************
#define SSI_CLK                 (1 << SSI_CLKPIN_POS)
#define SSI_CLK_PCTL            (SSI_CLKPIN_PCTL << (4 * SSI_CLKPIN_POS))


//*****************************************************************************
//
// This defines the SSI transmit pin that is being used by the boot loader.
//
//*****************************************************************************
#define SSI_TX                  (1 << SSI_MOSIPIN_POS)
#define SSI_TX_PCTL             (SSI_MOSIPIN_PCTL << 4 * SSI_MOSIPIN_POS)

//*****************************************************************************
//
// This defines the SSI receive pin that is being used by the boot loader.
//
//*****************************************************************************
#define SSI_RX                  (1 << SSI_MISOPIN_POS)
#define SSI_RX_PCTL             (SSI_MISOPIN_PCTL << (4 * SSI_MISOPIN_POS))

//*****************************************************************************
//
// This defines the combination of pins used to implement the SSI port used by
// the boot loader.
//
//*****************************************************************************
#define SSI_PINS                (SSI_CLK | SSI_TX | SSI_RX | SSI_CS)
#define SSI_PINS_PCTL           (SSI_CLK_PCTL | SSI_TX_PCTL | SSI_RX_PCTL |  \
                                 SSI_CS_PCTL)

//*****************************************************************************
//
// SSI Transport APIs
//
//*****************************************************************************
extern void SSISend(const uint8_t *pui8Data, uint32_t ui32Size);
extern void SSIReceive(uint8_t *pui8Data, uint32_t ui32Size);
extern void SSIFlush(void);

//*****************************************************************************
//
// Define the transport functions if the SSI port is being used.
//
//*****************************************************************************
#ifdef SSI_ENABLE_UPDATE
#define SendData                SSISend
#define FlushData               SSIFlush
#define ReceiveData             SSIReceive
#endif

#endif // __BL_SSI_H__
