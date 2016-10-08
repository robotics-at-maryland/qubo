//*****************************************************************************
//
// trf79x0_hw.h - Hardware Pin configuration for TRF79x0 ABP on
// Tiva C Series. Tailored for EK-TM4C1294XL.
//
// Copyright (c) 2014-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
//
//*****************************************************************************

#ifndef __TRF79X0_HW_H__
#define __TRF79X0_HW_H__

//*****************************************************************************
//
// Enable the TRF79x0 that will be used.
// Enabled = 1, Disabled = 0
//
//*****************************************************************************
#define RF_DAUGHTER_TRF7960         0
#define RF_DAUGHTER_TRF7970         1

//*****************************************************************************
//
// Select which boosterpack headers to use.
//
//*****************************************************************************
#define TRF79X0_USE_BOOSTERPACK_1       0
#define TRF79X0_USE_BOOSTERPACK_2       1

//*****************************************************************************
//
// Check for correct definition of RF_DAUGTHER_TRF79X0
//
//*****************************************************************************
#if (RF_DAUGHTER_TRF7960 && RF_DAUGHTER_TRF7970)
#error "Only one TRF79X0 can be defined at the same time."
#elif (!(RF_DAUGHTER_TRF7960 || RF_DAUGHTER_TRF7970))
#error "Define the TRF79X0 to be used, none currently defined."
#endif

//*****************************************************************************
//
// Check for correct definition of TRF79X0_USE_BOOSTERPACK_X
//
//*****************************************************************************
#if (TRF79X0_USE_BOOSTERPACK_1 && TRF79X0_USE_BOOSTERPACK_2)
#error "Both TRF79X0_USE_BOOSTERPACK_1 and TRF79X0_USE_BOOSTERPACK_2 are "
#error "defined. Please choose only one."
#elif (!(TRF79X0_USE_BOOSTERPACK_1 || TRF79X0_USE_BOOSTERPACK_2))
#error "Define the Boosterpack connection to be used, none currently defined."
#endif

//*****************************************************************************
//
// SSI Rate Deffinitions.
//
//*****************************************************************************
#define SSI_CLK_RATE            2000000
#define SSI_CLKS_PER_MS         (SSI_CLK_RATE / 1000)
#define STATUS_READS_PER_MS     (SSI_CLKS_PER_MS / 16)
#define SSI_NO_DATA              0

#if(TRF79X0_USE_BOOSTERPACK_1)
//*****************************************************************************
//
// BoosterPack 1 signal deffinitions.
//
//*****************************************************************************
#define TRF79X0_SSI_PERIPH      SYSCTL_PERIPH_SSI2
#define TRF79X0_SSI_BASE        SSI2_BASE

#define TRF79X0_CLK_BASE        GPIO_PORTD_BASE
#define TRF79X0_CLK_PERIPH      SYSCTL_PERIPH_GPIOD
#define TRF79X0_CLK_PIN         GPIO_PIN_3
#define TRF79X0_CLK_CONFIG      GPIO_PD3_SSI2CLK

#define TRF79X0_TX_BASE         GPIO_PORTD_BASE
#define TRF79X0_TX_PERIPH       SYSCTL_PERIPH_GPIOD
#define TRF79X0_TX_PIN          GPIO_PIN_1
#define TRF79X0_TX_CONFIG       GPIO_PD1_SSI2XDAT0

#define TRF79X0_RX_BASE         GPIO_PORTD_BASE
#define TRF79X0_RX_PERIPH       SYSCTL_PERIPH_GPIOD
#define TRF79X0_RX_PIN          GPIO_PIN_0
#define TRF79X0_RX_CONFIG       GPIO_PD0_SSI2XDAT1

#define TRF79X0_CS_BASE         GPIO_PORTB_BASE
#define TRF79X0_CS_PERIPH       SYSCTL_PERIPH_GPIOB
#define TRF79X0_CS_PIN          GPIO_PIN_2

#define TRF79X0_EN_BASE         GPIO_PORTB_BASE
#define TRF79X0_EN_PERIPH       SYSCTL_PERIPH_GPIOB
#define TRF79X0_EN_PIN          GPIO_PIN_3

#define TRF79X0_IRQ_BASE        GPIO_PORTC_BASE
#define TRF79X0_IRQ_PERIPH      SYSCTL_PERIPH_GPIOC
#define TRF79X0_IRQ_PIN         GPIO_PIN_7
#define TRF79X0_IRQ_INT         INT_GPIOC

#elif (TRF79X0_USE_BOOSTERPACK_2)
//*****************************************************************************
//
// Boosterpack 2 signal deffinitions.
//
//*****************************************************************************
#define TRF79X0_SSI_PERIPH      SYSCTL_PERIPH_SSI3
#define TRF79X0_SSI_BASE        SSI3_BASE

#define TRF79X0_CLK_BASE        GPIO_PORTQ_BASE
#define TRF79X0_CLK_PERIPH      SYSCTL_PERIPH_GPIOQ
#define TRF79X0_CLK_PIN         GPIO_PIN_0
#define TRF79X0_CLK_CONFIG      GPIO_PQ0_SSI3CLK

#define TRF79X0_TX_BASE         GPIO_PORTQ_BASE
#define TRF79X0_TX_PERIPH       SYSCTL_PERIPH_GPIOQ
#define TRF79X0_TX_PIN          GPIO_PIN_2
#define TRF79X0_TX_CONFIG       GPIO_PQ2_SSI3XDAT0

#define TRF79X0_RX_BASE         GPIO_PORTQ_BASE
#define TRF79X0_RX_PERIPH       SYSCTL_PERIPH_GPIOQ
#define TRF79X0_RX_PIN          GPIO_PIN_3
#define TRF79X0_RX_CONFIG       GPIO_PQ3_SSI3XDAT1

#define TRF79X0_CS_BASE         GPIO_PORTN_BASE
#define TRF79X0_CS_PERIPH       SYSCTL_PERIPH_GPION
#define TRF79X0_CS_PIN          GPIO_PIN_5

#define TRF79X0_EN_BASE         GPIO_PORTN_BASE
#define TRF79X0_EN_PERIPH       SYSCTL_PERIPH_GPION
#define TRF79X0_EN_PIN          GPIO_PIN_4

#define TRF79X0_IRQ_BASE        GPIO_PORTP_BASE
#define TRF79X0_IRQ_PERIPH      SYSCTL_PERIPH_GPIOP
#define TRF79X0_IRQ_PIN         GPIO_PIN_4
#define TRF79X0_IRQ_INT         INT_GPIOP0
#endif

//
// These Signals are not used on the dlp7970abp board, but must be defined for
// code compatibility with the TI produced trf7970atb EM header booster pack.
//
#define TRF79X0_EN2_BASE        0
#define TRF79X0_EN2_PERIPH      0
#define TRF79X0_EN2_PIN         0
#define TRF79X0_ASKOK_BASE      0
#define TRF79X0_ASKOK_PERIPH    0
#define TRF79X0_ASKOK_PIN       0
#define TRF79X0_MOD_BASE        0
#define TRF79X0_MOD_PERIPH      0
#define TRF79X0_MOD_PIN         0

//*****************************************************************************
//
// Macro for IRQ pin from TRF79x0 -> Board
// left in this format for cross platform compatibility.
//
//*****************************************************************************
#define IRQ_IS_SET()        GPIOPinRead(TRF79X0_IRQ_BASE, TRF79X0_IRQ_PIN)

#endif // __TRF79X0_HW_H__
