//*****************************************************************************
//
// aes128_cmac.c - Simple AES128 CMAC demo.
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
#include "driverlib/interrupt.h"
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
//! <h1>AES128 CMAC Demo (aes128_cmac)</h1>
//!
//! Simple demo showing an authentication operation using the AES128 module in
//! CMAC mode.  A series of test vectors are authenticated.
//!
//! This module is also capable of CBC-MAC mode, but this has been determined
//! to be insecure when using variable message lengths.  CMAC is now 
//! recommended instead by NIST.
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
// Test cases from the NIST SP 800-38B document.
// The data in these test cases have been modified to be in big endian 
// format as required by the AES module.  This was done to simplify writes
// and comparisons.  When operations need to be performed on the data, the
// endianness is swapped.
//
//*****************************************************************************
typedef struct AESTestVectorStruct
{
    uint32_t ui32Length;
    uint32_t pui32Message[16];
    uint32_t pui32Tag[4];
}
tAESCMACTestVector;

tAESCMACTestVector g_psAESCMACTestVectors[] =
{
    //
    // Test Case #1
    // Empty message check.  Since there is no message, it must be
    // padded with a one and 127 zeros.  Also, a zero cannot be
    // written into the length register in this mode, so we just 
    // write a 1 as the length to signify an incomplete block.
    // Any value from 1 to 15 would have worked in this case.
    // Incomplete blocks are XOR'd with subkey2 rather than subkey1.
    // 
    {
        1,
        { 0x00000080, 0x00000000, 0x00000000, 0x00000000 },
        { 0x29691dbb, 0x283759e9, 0x127da37f, 0x4667759b }
    },

    //
    // Test Case #2
    // This is the first complete block.  It is XOR'd with subkey1.
    //
    {
        16,
        { 0xe2bec16b, 0x969f402e, 0x117e3de9, 0x2a179373 },
        { 0xb4160a07, 0x44414d6b, 0x9ddd9bf7, 0x7c284ad0 }
    },
    
    //
    // Test Case #3
    // Since the message is not a multiple of 128 bits, there must
    // be padding appended to the end of the message.  This padding
    // is a one followed by 63 zeros.
    //
    {
        40,
        { 0xe2bec16b, 0x969f402e, 0x117e3de9, 0x2a179373,
          0x578a2dae, 0x9cac031e, 0xac6fb79e, 0x518eaf45,
          0x461cc830, 0x11e45ca3, 0x00000080, 0x00000000 },
        { 0x4767a6df, 0x30e69ade, 0x6132ca30, 0x27c89714 }
    },

    //
    // Test Case #4
    //
    {
        64,
        { 0xe2bec16b, 0x969f402e, 0x117e3de9, 0x2a179373,
          0x578a2dae, 0x9cac031e, 0xac6fb79e, 0x518eaf45,
          0x461cc830, 0x11e45ca3, 0x19c1fbe5, 0xef520a1a,
          0x45249ff6, 0x179b4fdf, 0x7b412bad, 0x10376ce6 },
        { 0xbfbef051, 0x929d3b7e, 0x177449fc, 0xfe3c3679 }
    }
};

//
// The following key is used for all of the above test cases.
//
uint32_t g_pui32AES128Key[4] =
{
    0x16157e2b, 0xa6d2ae28, 0x8815f7ab, 0x3c4fcf09
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
// Switch the endianness of the data array.
//
//*****************************************************************************
void
EndiannessSwap(uint32_t *pui32Input, uint32_t *pui32Output, 
              uint32_t ui32Length)
{
    uint32_t ui32Count;
    
    //
    // For each word, swap the endianness.
    //
    for(ui32Count = 0; ui32Count < ui32Length; ui32Count++)
    {
        pui32Output[ui32Count] = ((pui32Input[ui32Count] & 0x000000ff) << 24) |
                                 ((pui32Input[ui32Count] & 0x0000ff00) << 8) |
                                 ((pui32Input[ui32Count] & 0x00ff0000) >> 8) |
                                 ((pui32Input[ui32Count] & 0xff000000) >> 24);   

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
// Perform an ECB encryption operation.
//
//*****************************************************************************
bool
AES128ECBEncrypt(uint32_t *pui32Src, uint32_t *pui32Dst, uint32_t *pui32Key,
                 uint32_t ui32Length)
{
    //
    // Perform a soft reset.
    //
    ROM_AESReset(AES_BASE);

    //
    // Configure the AES module.
    //
    ROM_AESConfigSet(AES_BASE, (AES_CFG_KEY_SIZE_128BIT | AES_CFG_DIR_ENCRYPT |
                                AES_CFG_MODE_ECB));

    //
    // Write the key.
    //
    ROM_AESKey1Set(AES_BASE, pui32Key, AES_CFG_KEY_SIZE_128BIT);

    //
    // Perform the encryption.
    //
    ROM_AESDataProcess(AES_BASE, pui32Src, pui32Dst, ui32Length);

    return(true);
}

//*****************************************************************************
//
// Generate a CMAC subkey.
//
//*****************************************************************************
bool
AES128CMACSubkeyGet(uint32_t *pui32Key, uint32_t *pui32Input, 
                    uint32_t *pui32Subkey)
{
    uint32_t pui32Output[4];
    uint32_t pui32SwappedInput[4];
    int32_t i32Idx;   
    bool bCarry;

    //
    // If the MSB of the input is 0, then the subkey is just left shifted.
    // If the MSB of the input is 1, then the subkey is left shifted and
    // XOR'd with a constant.  First swap the endianness to big endian
    // to make the math easier.
    //
    EndiannessSwap(pui32Input, pui32SwappedInput, 4);

    //
    // Shift each word in the 128 bits.  Make sure to carry the left
    // shifted bits.
    //
    bCarry = false;
    for(i32Idx = 3; i32Idx >= 0; i32Idx--)
    {
        //
        // Shift the word.
        //
        pui32Output[i32Idx] = pui32SwappedInput[i32Idx] << 1;

        //
        // If there was a carry from the previous word.
        //
        if(bCarry)
        {
            pui32Output[i32Idx] |= 0x1;
            bCarry = false;
        }

        //
        // Check to see if we need to carry to the next word.
        //
        if(pui32SwappedInput[i32Idx] & 0x80000000)
        {   
            bCarry = true;
        }
    }
   
    //
    // Swap the endianness back to little endian.
    //
    EndiannessSwap(pui32Output, pui32Subkey, 4);

    //
    // XOR in the Rb constant if the MSB is 1.
    //
    if(pui32SwappedInput[0] & 0x80000000)
    {
        pui32Subkey[3] ^= 0x87000000;
    }

    return true;
}

//*****************************************************************************
//
// Perform an encryption operation.
//
//*****************************************************************************
bool
AES128CMACAuth(uint32_t *pui32Src, uint32_t *pui32Key, uint32_t *pui32Tag, 
               uint32_t ui32Length, bool bUseDMA)
{
    uint32_t pui32Subkey1[4];
    uint32_t pui32Subkey2[4];
    uint32_t pui32Zero[4];
    uint32_t pui32EncZero[4];

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
    // Calculate the first subkey.  First, encrypt a zero string.
    //
    pui32Zero[0] = 0x00000000;
    pui32Zero[1] = 0x00000000;
    pui32Zero[2] = 0x00000000;
    pui32Zero[3] = 0x00000000;

    //
    // Encrypt the zero string. 
    //
    AES128ECBEncrypt(pui32Zero, pui32EncZero, pui32Key, 16);

    //
    // Get the first subkey.
    //
    AES128CMACSubkeyGet(pui32Key, pui32EncZero, pui32Subkey1);

    //
    // Get the second subkey.
    //
    AES128CMACSubkeyGet(pui32Key, pui32Subkey1, pui32Subkey2);

    //
    // Enable all interrupts.
    //
    ROM_AESIntEnable(AES_BASE, (AES_INT_CONTEXT_IN | AES_INT_CONTEXT_OUT |
                                AES_INT_DATA_IN | AES_INT_DATA_OUT));

    //
    // Configure the AES module.
    //
    ROM_AESConfigSet(AES_BASE, (AES_CFG_KEY_SIZE_128BIT | AES_CFG_DIR_ENCRYPT |
                                AES_CFG_MODE_CBCMAC));

    //
    // Write the key.
    //
    ROM_AESKey1Set(AES_BASE, pui32Key, AES_CFG_KEY_SIZE_128BIT);

    //
    // Write the first subkey.
    //
    ROM_AESKey2Set(AES_BASE, pui32Subkey1, AES_CFG_KEY_SIZE_128BIT);

    //
    // Write the second subkey.
    //
    ROM_AESKey3Set(AES_BASE, pui32Subkey2);

    //
    // Write the IV with zeroes.
    //
    ROM_AESIVSet(AES_BASE, pui32Zero);

    //
    // Depending on the argument, perform the encryption
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
        // Write the length registers to start the process.
        //
        ROM_AESLengthSet(AES_BASE, (uint64_t)ui32Length);
        
        //
        // Enable the DMA channel to start the transfer.  This must be done after
        // writing the length to prevent data from copying before the context is 
        // truly ready.
        // 
        ROM_uDMAChannelEnable(UDMA_CH14_AES0DIN);

        //
        // Enable DMA requests
        //
        ROM_AESDMAEnable(AES_BASE, AES_DMA_DATA_IN);

        //
        // Wait for the data in DMA done interrupt.
        //
        while(!g_bDataInDMADoneIntFlag)
        {
        }

        //
        // Read out the tag.
        //
        ROM_AESTagRead(AES_BASE, pui32Tag);
    }
    else
    {
        //
        // Perform the authentication.
        //
        ROM_AESDataAuth(AES_BASE, pui32Src, ui32Length, pui32Tag);
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
// This example authenticates blocks of plaintext using AES128 in CMAC mode.  
// It does the encryption first without uDMA and then with uDMA.  The results
// are checked after each operation.
//
//*****************************************************************************
int
main(void)
{
    uint32_t *pui32ExpTag, *pui32Message;
    uint32_t ui32Errors, ui32Idx, ui32Length, pui32Tag[4], ui32SysClock;
    uint8_t ui8Vector;
    
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
    for(ui32Idx = 0; ui32Idx < 4; ui32Idx++)
    {
        pui32Tag[ui32Idx] = 0;
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
    UARTprintf("Starting AES128 CMAC encryption demo.\n");
    
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
    // Loop through all the given vectors.
    //
    for(ui8Vector = 0;
        (ui8Vector < 
         (sizeof(g_psAESCMACTestVectors) / sizeof(g_psAESCMACTestVectors[0]))) &&
        (ui32Errors == 0);
        ui8Vector++)
    {
        UARTprintf("Starting vector #%d\n", ui8Vector);

        //
        // Get the current vector's data members.
        //
        ui32Length = g_psAESCMACTestVectors[ui8Vector].ui32Length;
        pui32Message = g_psAESCMACTestVectors[ui8Vector].pui32Message;
        pui32ExpTag = g_psAESCMACTestVectors[ui8Vector].pui32Tag;
        
        //
        // Perform the encryption without uDMA.
        //
        UARTprintf("Performing encryption without uDMA.\n");
        AES128CMACAuth(pui32Message, g_pui32AES128Key,
                       pui32Tag, ui32Length, false);

        //
        // Check the result.
        //
        for(ui32Idx = 0; ui32Idx < 4; ui32Idx++)
        {
            if(pui32Tag[ui32Idx] != pui32ExpTag[ui32Idx])
            {
                UARTprintf("Tag mismatch on word %d. Exp: 0x%x, Act: "
                           "0x%x\n", ui32Idx, pui32ExpTag[ui32Idx],
                           pui32Tag[ui32Idx]);
                ui32Errors |= (ui32Idx << 16) | 0x00000002;
            }
        }

        //
        // Clear the array containing the tag.
        //
        for(ui32Idx = 0; ui32Idx < 4; ui32Idx++)
        {
            pui32Tag[ui32Idx] = 0;
        }

        //
        // Only use DMA with the vectors that have data.
        //
        if(ui32Length != 0)
        {
            //
            // Perform the encryption with uDMA.
            //
            UARTprintf("Performing encryption with uDMA.\n");
            AES128CMACAuth(pui32Message, g_pui32AES128Key,
                           pui32Tag, ui32Length, true);

            //
            // Check the result.
            //
            for(ui32Idx = 0; ui32Idx < 4; ui32Idx++)
            {
                if(pui32Tag[ui32Idx] != pui32ExpTag[ui32Idx])
                {
                    UARTprintf("Tag mismatch on word %d. Exp: 0x%x, Act: "
                               "0x%x\n", ui32Idx, pui32ExpTag[ui32Idx],
                               pui32Tag[ui32Idx]);
                    ui32Errors |= (ui32Idx << 16) | 0x00000004;
                }
            }

            //
            // Clear the array containing the tag.
            //
            for(ui32Idx = 0; ui32Idx < 4; ui32Idx++)
            {
                pui32Tag[ui32Idx] = 0;
            }
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
