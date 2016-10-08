//*****************************************************************************
//
// aes128_cbc_decrypt.c - Simple AES128 CBC decryption demo.
//
// Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the EK-TM4C129EXL Firmware Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_aes.h"
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "driverlib/aes.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "driverlib/udma.h"
#include "drivers/pinout.h"
#include "utils/uartstdio.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>AES128 CBC Decryption Demo (aes128_cbc_decrypt)</h1>
//!
//! Simple demo showing an decryption operation using the AES128 module in
//! CBC mode.  A number of blocks of data are decrypted.
//!
//! Please note that the use of interrupts and uDMA is not required for the
//! operation of the module.  It is only done for demonstration purposes.
//
//*****************************************************************************

//*****************************************************************************
//
// Configuration defines.
//
//*****************************************************************************
#define CCM_LOOP_TIMEOUT        500000

//*****************************************************************************
//
// The DMA control structure table.
//
//*****************************************************************************
#if defined(ewarm)
#pragma data_alignment=1024
tDMAControlTable g_psDMAControlTable[64];
#elif defined(ccs)
#pragma DATA_ALIGN(g_psDMAControlTable, 1024)
tDMAControlTable g_psDMAControlTable[64];
#else
tDMAControlTable g_psDMAControlTable[64] __attribute__((aligned(1024)));
#endif

//*****************************************************************************
//
// Sample plaintext, ciphertext, and key from the NIST SP 800-38A document.
//
//*****************************************************************************
uint32_t g_pui32AES128PlainText[16] =
{
    0xe2bec16b, 0x969f402e, 0x117e3de9, 0x2a179373,
    0x578a2dae, 0x9cac031e, 0xac6fb79e, 0x518eaf45,
    0x461cc830, 0x11e45ca3, 0x19c1fbe5, 0xef520a1a,
    0x45249ff6, 0x179b4fdf, 0x7b412bad, 0x10376ce6
};

uint32_t g_pui32AES128Key[4] =
{
    0x16157e2b, 0xa6d2ae28, 0x8815f7ab, 0x3c4fcf09
};

uint32_t g_pui32AES128IV[4] = 
{
    0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c
};

uint32_t g_pui32AES128CipherText[16] =
{
    0xacab4976, 0x46b21981, 0x9b8ee9ce, 0x7d19e912,
    0x9bcb8650, 0xee197250, 0x3a11db95, 0xb2787691,
    0xb8d6be73, 0x3b74c1e3, 0x9ee61671, 0x16952222,
    0xa1caf13f, 0x09ac1f68, 0x30ca0e12, 0xa7e18675
};

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
}
#endif

//*****************************************************************************
//
// Round up length to nearest 16 byte boundary.  This is needed because all
// four data registers must be written at once.  This is handled in the AES
// driver, but if using uDMA, the length must rounded up.
//
//*****************************************************************************
uint32_t
LengthRoundUp(uint32_t ui32Length)
{
    uint32_t ui32Remainder;

    ui32Remainder = ui32Length % 16;
    if(ui32Remainder == 0)
    {
        return(ui32Length);
    }
    else
    {
        return(ui32Length + (16 - ui32Remainder));
    }
}

//*****************************************************************************
//
// The AES interrupt handler and interrupt flags.
//
//*****************************************************************************
static volatile bool g_bContextInIntFlag;
static volatile bool g_bDataInIntFlag;
static volatile bool g_bContextOutIntFlag;
static volatile bool g_bDataOutIntFlag;
static volatile bool g_bContextInDMADoneIntFlag;
static volatile bool g_bDataInDMADoneIntFlag;
static volatile bool g_bContextOutDMADoneIntFlag;
static volatile bool g_bDataOutDMADoneIntFlag;

void
AESIntHandler(void)
{
    uint32_t ui32IntStatus;

    //
    // Read the AES masked interrupt status.
    //
    ui32IntStatus = ROM_AESIntStatus(AES_BASE, true);

    //
    // Print a different message depending on the interrupt source.
    //
    if(ui32IntStatus & AES_INT_CONTEXT_IN)
    {
        ROM_AESIntDisable(AES_BASE, AES_INT_CONTEXT_IN);
        g_bContextInIntFlag = true;
        UARTprintf(" Context input registers are ready.\n");
    }
    if(ui32IntStatus & AES_INT_DATA_IN)
    {
        ROM_AESIntDisable(AES_BASE, AES_INT_DATA_IN);
        g_bDataInIntFlag = true;
        UARTprintf(" Data FIFO is ready to receive data.\n");
    }
    if(ui32IntStatus & AES_INT_CONTEXT_OUT)
    {
        ROM_AESIntDisable(AES_BASE, AES_INT_CONTEXT_OUT);
        g_bContextOutIntFlag = true;
        UARTprintf(" Context output registers are ready.\n");
    }
    if(ui32IntStatus & AES_INT_DATA_OUT)
    {
        ROM_AESIntDisable(AES_BASE, AES_INT_DATA_OUT);
        g_bDataOutIntFlag = true;
        UARTprintf(" Data FIFO is ready to provide data.\n");
    }
    if(ui32IntStatus & AES_INT_DMA_CONTEXT_IN)
    {
        ROM_AESIntClear(AES_BASE, AES_INT_DMA_CONTEXT_IN);
        g_bContextInDMADoneIntFlag = true;
        UARTprintf(" DMA completed a context write to the internal\n");
        UARTprintf(" registers.\n");
    }
    if(ui32IntStatus & AES_INT_DMA_DATA_IN)
    {
        ROM_AESIntClear(AES_BASE, AES_INT_DMA_DATA_IN);
        g_bDataInDMADoneIntFlag = true;
        UARTprintf(" DMA has written the last word of input data to\n");
        UARTprintf(" the internal FIFO of the engine.\n");
    }
    if(ui32IntStatus & AES_INT_DMA_CONTEXT_OUT)
    {
        ROM_AESIntClear(AES_BASE, AES_INT_DMA_CONTEXT_OUT);
        g_bContextOutDMADoneIntFlag = true;
        UARTprintf(" DMA completed the output context movement from\n");
        UARTprintf(" the internal registers.\n");
    }
    if(ui32IntStatus & AES_INT_DMA_DATA_OUT)
    {
        ROM_AESIntClear(AES_BASE, AES_INT_DMA_DATA_OUT);
        g_bDataOutDMADoneIntFlag = true;
        UARTprintf(" DMA has written the last word of process result.\n");
    }
}

//*****************************************************************************
//
// Perform an decryption operation.
//
//*****************************************************************************
bool
AES128CBCDecrypt(uint32_t *pui32Src, uint32_t *pui32Dst, uint32_t *pui32Key,
                 uint32_t *pui32IV, uint32_t ui32Length, bool bUseDMA)
{
    //
    // Perform a soft reset.
    //
    ROM_AESReset(AES_BASE);

    //
    // Clear the interrupt flags.
    //
    g_bContextInIntFlag = false;
    g_bDataInIntFlag = false;
    g_bContextOutIntFlag = false;
    g_bDataOutIntFlag = false;
    g_bContextInDMADoneIntFlag = false;
    g_bDataInDMADoneIntFlag = false;
    g_bContextOutDMADoneIntFlag = false;
    g_bDataOutDMADoneIntFlag = false;

    //
    // Enable all interrupts.
    //
    ROM_AESIntEnable(AES_BASE, (AES_INT_CONTEXT_IN | AES_INT_CONTEXT_OUT |
                                AES_INT_DATA_IN | AES_INT_DATA_OUT));

    //
    // Configure the AES module.
    //
    ROM_AESConfigSet(AES_BASE, (AES_CFG_KEY_SIZE_128BIT | AES_CFG_DIR_DECRYPT |
                                AES_CFG_MODE_CBC));

    //
    // Write the initial value.
    //
    AESIVSet(AES_BASE, pui32IV);

    //
    // Write the key.
    //
    ROM_AESKey1Set(AES_BASE, pui32Key, AES_CFG_KEY_SIZE_128BIT);

    //
    // Depending on the argument, perform the decryption
    // with or without uDMA.
    //
    if(bUseDMA)
    {
        //
        // Enable DMA interrupts.
        //
        ROM_AESIntEnable(AES_BASE, (AES_INT_DMA_CONTEXT_IN |
                                    AES_INT_DMA_DATA_IN |
                                    AES_INT_DMA_CONTEXT_OUT |
                                    AES_INT_DMA_DATA_OUT));

        //
        // Setup the DMA module to copy data in.
        //
        ROM_uDMAChannelAssign(UDMA_CH14_AES0DIN);
        ROM_uDMAChannelAttributeDisable(UDMA_CH14_AES0DIN,
                                        UDMA_ATTR_ALTSELECT |
                                        UDMA_ATTR_USEBURST |
                                        UDMA_ATTR_HIGH_PRIORITY |
                                        UDMA_ATTR_REQMASK);
        ROM_uDMAChannelControlSet(UDMA_CH14_AES0DIN | UDMA_PRI_SELECT,
                                  UDMA_SIZE_32 | UDMA_SRC_INC_32 |
                                  UDMA_DST_INC_NONE | UDMA_ARB_4 |
                                  UDMA_DST_PROT_PRIV);
        ROM_uDMAChannelTransferSet(UDMA_CH14_AES0DIN | UDMA_PRI_SELECT,
                                   UDMA_MODE_BASIC, (void *)pui32Src,
                                   (void *)(AES_BASE + AES_O_DATA_IN_0),
                                   LengthRoundUp(ui32Length) / 4);
        UARTprintf("Data in DMA request enabled.\n");

        //
        // Setup the DMA module to copy the data out.
        //
        ROM_uDMAChannelAssign(UDMA_CH15_AES0DOUT);
        ROM_uDMAChannelAttributeDisable(UDMA_CH15_AES0DOUT,
                                        UDMA_ATTR_ALTSELECT |
                                        UDMA_ATTR_USEBURST |
                                        UDMA_ATTR_HIGH_PRIORITY |
                                        UDMA_ATTR_REQMASK);
        ROM_uDMAChannelControlSet(UDMA_CH15_AES0DOUT | UDMA_PRI_SELECT,
                                  UDMA_SIZE_32 | UDMA_SRC_INC_NONE |
                                  UDMA_DST_INC_32 | UDMA_ARB_4 |
                                  UDMA_SRC_PROT_PRIV);
        ROM_uDMAChannelTransferSet(UDMA_CH15_AES0DOUT | UDMA_PRI_SELECT,
                                   UDMA_MODE_BASIC,
                                   (void *)(AES_BASE + AES_O_DATA_IN_0),
                                   (void *)pui32Dst,
                                   LengthRoundUp(ui32Length) / 4);
        UARTprintf("Data out DMA request enabled.\n");

        //
        // Write the length registers to start the process.
        //
        ROM_AESLengthSet(AES_BASE, (uint64_t)ui32Length);

        //
        // Enable the DMA channels to start the transfers.  This must be done after
        // writing the length to prevent data from copying before the context is 
        // truly ready.
        // 
        ROM_uDMAChannelEnable(UDMA_CH14_AES0DIN);
        ROM_uDMAChannelEnable(UDMA_CH15_AES0DOUT);

        //
        // Enable DMA requests
        //
        ROM_AESDMAEnable(AES_BASE, AES_DMA_DATA_IN | AES_DMA_DATA_OUT);

        //
        // Wait for the data in DMA done interrupt.
        //
        while(!g_bDataInDMADoneIntFlag)
        {
        }

        //
        // Wait for the data out DMA done interrupt.
        //
        while(!g_bDataOutDMADoneIntFlag)
        {
        }
    }
    else
    {
        //
        // Perform the decryption.
        //
        ROM_AESDataProcess(AES_BASE, pui32Src, pui32Dst, ui32Length);
    }

    return(true);
}

//*****************************************************************************
//
// Initialize the AES and CCM modules.
//
//*****************************************************************************
bool
AESInit(void)
{
    uint32_t ui32Loop;

    //
    // Check that the CCM peripheral is present.
    //
    if(!ROM_SysCtlPeripheralPresent(SYSCTL_PERIPH_CCM0))
    {
        UARTprintf("No CCM peripheral found!\n");

        //
        // Return failure.
        //
        return(false);
    }

    //
    // The hardware is available, enable it.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_CCM0);

    //
    // Wait for the peripheral to be ready.
    //
    ui32Loop = 0;
    while(!ROM_SysCtlPeripheralReady(SYSCTL_PERIPH_CCM0))
    {
        //
        // Increment our poll counter.
        //
        ui32Loop++;

        if(ui32Loop > CCM_LOOP_TIMEOUT)
        {
            //
            // Timed out, notify and spin.
            //
            UARTprintf("Time out on CCM ready after enable.\n");

            //
            // Return failure.
            //
            return(false);
        }
    }

    //
    // Reset the peripheral to ensure we are starting from a known condition.
    //
    ROM_SysCtlPeripheralReset(SYSCTL_PERIPH_CCM0);

    //
    // Wait for the peripheral to be ready again.
    //
    ui32Loop = 0;
    while(!ROM_SysCtlPeripheralReady(SYSCTL_PERIPH_CCM0))
    {
        //
        // Increment our poll counter.
        //
        ui32Loop++;

        if(ui32Loop > CCM_LOOP_TIMEOUT)
        {
            //
            // Timed out, spin.
            //
            UARTprintf("Time out on CCM ready after reset.\n");

            //
            // Return failure.
            //
            return(false);
        }
    }

    //
    // Return initialization success.
    //
    return(true);
}

//*****************************************************************************
//
// Configure the UART and its pins.  This must be called before UARTprintf().
//
//*****************************************************************************
void
ConfigureUART(void)
{
    //
    // Enable UART0
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Use the internal 16MHz oscillator as the UART clock source.
    //
    ROM_UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, 16000000);
}

//*****************************************************************************
//
// This example decrypts blocks of plaintext using AES128 in CBC mode.  It
// does the decryption first without uDMA and then with uDMA.  The results
// are checked after each operation.
//
//*****************************************************************************
int
main(void)
{
    uint32_t pui32PlainText[16], ui32Errors, ui32Idx, ui32SysClock;
    
    //
    // Run from the PLL at 120 MHz.
    //
    ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                           SYSCTL_OSC_MAIN |
                                           SYSCTL_USE_PLL |
                                           SYSCTL_CFG_VCO_480), 120000000);

    //
    // Configure the device pins.
    //
    PinoutSet(false, false);

    //
    // Initialize local variables.
    //
    ui32Errors = 0;
    for(ui32Idx = 0; ui32Idx < 16; ui32Idx++)
    {
        pui32PlainText[ui32Idx] = 0;
    }

    //
    // Enable stacking for interrupt handlers.  This allows floating-point
    // instructions to be used within interrupt handlers, but at the expense of
    // extra stack usage.
    //
    ROM_FPUStackingEnable();

    //
    // Enable AES interrupts.
    //
    ROM_IntEnable(INT_AES0);

    //
    // Enable debug output on UART0 and print a welcome message.
    //
    ConfigureUART();
	UARTprintf("\033[2J\033[H");
    UARTprintf("Starting AES128 CBC decryption demo.\n");

    //
    // Enable the uDMA module.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UDMA);

    //
    // Setup the control table.
    //
    ROM_uDMAEnable();
    ROM_uDMAControlBaseSet(g_psDMAControlTable);

    //
    // Initialize the CCM and AES modules.
    //
    if(!AESInit())
    {
        UARTprintf("Initialization of the AES module failed.\n");
        ui32Errors |= 0x00000001;
    }

    //
    // Perform the decryption without uDMA.
    //
    UARTprintf("Performing decryption without uDMA.\n");
    AES128CBCDecrypt(g_pui32AES128CipherText, pui32PlainText, g_pui32AES128Key,
                     g_pui32AES128IV, 64, false);

    //
    // Check the result.
    //
    for(ui32Idx = 0; ui32Idx < 16; ui32Idx++)
    {
        if(pui32PlainText[ui32Idx] != g_pui32AES128PlainText[ui32Idx])
        {
            UARTprintf("Plaintext mismatch on word %d. Exp: 0x%x, Act: "
                       "0x%x\n", ui32Idx, g_pui32AES128PlainText[ui32Idx],
                       pui32PlainText[ui32Idx]);
            ui32Errors |= (ui32Idx << 16) | 0x00000002;
        }
    }

    //
    // Clear the array containing the plaintext.
    //
    for(ui32Idx = 0; ui32Idx < 16; ui32Idx++)
    {
        pui32PlainText[ui32Idx] = 0;
    }

    //
    // Perform the decryption with uDMA.
    //
    UARTprintf("Performing decryption with uDMA.\n");
    AES128CBCDecrypt(g_pui32AES128CipherText, pui32PlainText, g_pui32AES128Key,
                     g_pui32AES128IV, 64, true);

    //
    // Check the result.
    //
    for(ui32Idx = 0; ui32Idx < 16; ui32Idx++)
    {
        if(pui32PlainText[ui32Idx] != g_pui32AES128PlainText[ui32Idx])
        {
            UARTprintf("Plaintext mismatch on word %d. Exp: 0x%x, Act: "
                       "0x%x\n", ui32Idx, g_pui32AES128PlainText[ui32Idx],
                       pui32PlainText[ui32Idx]);
            ui32Errors |= (ui32Idx << 16) | 0x00000004;
        }
    }

    //
    // Finished.
    //
    if(ui32Errors)
    {
        UARTprintf("Demo failed with error code 0x%x.\n", ui32Errors);
        LEDWrite(CLP_D3 | CLP_D4, CLP_D4);
    }
    else
    {
        UARTprintf("Demo completed successfully.\n");
        LEDWrite(CLP_D3 | CLP_D4, CLP_D3);
    }

    while(1)
    {

    }
}
