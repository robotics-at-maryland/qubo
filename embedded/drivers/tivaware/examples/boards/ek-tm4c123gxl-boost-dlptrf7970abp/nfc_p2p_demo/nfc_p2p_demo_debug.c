//*****************************************************************************
//
// nfc_p2p_demo_debug.c - contains debug over UART functions
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
// This is part of revision 2.1.3.156 of the EK-TM4C123GXL Firmware Package.
//
//*****************************************************************************
#include <stdint.h>
#include <stdbool.h>
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "nfclib/nfc_p2p.h"

//*****************************************************************************
//
// Print Header Debug Info
//
//*****************************************************************************
void
DebugHeader(sNDEFMessageData sNDEFMessage)
{
    uint32_t x=0;

    UARTprintf("===== Message Header Begin =====\n");
    UARTprintf("\nDecoded Info:\n");
    UARTprintf("    StatusByte:\n");
    UARTprintf("        MB : 0x%x \n",sNDEFMessage.sStatusByte.MB);
    UARTprintf("        ME : 0x%x \n",sNDEFMessage.sStatusByte.ME);
    UARTprintf("        CF : 0x%x \n",sNDEFMessage.sStatusByte.CF);
    UARTprintf("        SR : 0x%x \n",sNDEFMessage.sStatusByte.SR);
    UARTprintf("        IL : 0x%x \n",sNDEFMessage.sStatusByte.IL);
    UARTprintf("        TNF: 0x%x \n",sNDEFMessage.sStatusByte.TNF);
    UARTprintf("    TypeLength:     0x%x \n",    sNDEFMessage.ui8TypeLength);
    UARTprintf("    PayloadLength:  0x%x , %d\n",sNDEFMessage.ui32PayloadLength,
                                                sNDEFMessage.ui32PayloadLength);
    UARTprintf("    IDLength:       0x%x \n",    sNDEFMessage.ui8IDLength);
    UARTprintf("    Type:           ");
    for(x=0;x<NDEF_TYPE_MAXSIZE;x++)
        {
            UARTprintf("%c",sNDEFMessage.pui8Type[x]);
        }
    UARTprintf("\n");
    UARTprintf("    ID:             0x");
    for(x=0;x<NDEF_ID_MAXSIZE;x++)
        {
            UARTprintf("%c",sNDEFMessage.pui8ID[x]);
        }
    UARTprintf("\n");
    UARTprintf("    PayloadPtr:     0x%x \n",sNDEFMessage.pui8PayloadPtr);
    UARTprintf("===== Message Header End =====\n\n");

    #ifdef DEBUG_PRINT
    UARTprintf("\n=====Payload Begin =====\n");
    //
    // Wait for the UART to catch up.
    //
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    for(x=0;x<sNDEFMessage.ui32PayloadLength;x++)
    {
        if(x%20==0)
            {
                #if defined(UART_BUFFERED)
                UARTFlushTx(false);
                #endif
            }
        UARTprintf("        Payload[%d]=0x%x, '%c' \n",x,
                    *(sNDEFMessage.pui8PayloadPtr + x),
                    *(sNDEFMessage.pui8PayloadPtr + x));
    }

    UARTprintf("\n=====Payload End=====\n");
    #endif //DEBUG_PRINT

    //
    // Wait for the UART to catch up.
    //
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif

    return;
}

//*****************************************************************************
//
// Print Text Record Debug Info
//
//*****************************************************************************
void
DebugTextRecord(sNDEFTextRecord sNDEFText)
{
    uint32_t x=0;

    UARTprintf("    Tag is Text Record \n");
    UARTprintf("    Text:'");
    for(x=0;x<sNDEFText.ui32TextLength;x++)
    {
        if(x%20==0)
            {
                #if defined(UART_BUFFERED)
                UARTFlushTx(false);
                #endif
            }
        UARTprintf("%c",sNDEFText.pui8Text[x]);
    }
    UARTprintf("'\n");
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    return;
}

//*****************************************************************************
//
// Print URI Record Debug Info
//
//*****************************************************************************
void
DebugURIRecord(sNDEFURIRecord sNDEFURI)
{
    uint32_t x=0;

    UARTprintf("    Tag is URI Record \n");
    UARTprintf("    URI IDCode: 0x%x\n",sNDEFURI.eIDCode);
    UARTprintf("    URI:'");
    for(x=0;x<sNDEFURI.ui32URILength;x++)
    {
        if(x%20==0)
            {
                #if defined(UART_BUFFERED)
                UARTFlushTx(false);
                #endif
            }
        UARTprintf("%c",sNDEFURI.puiUTF8String[x]);
    }
    UARTprintf("'\n");
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    return;
}

//*****************************************************************************
//
// Print Smart Poster Debug Info
//
//*****************************************************************************
void
DebugSmartPosterRecord(sNDEFSmartPosterRecord sNDEFSmartPoster)
{
    uint32_t x=0;

    UARTprintf("    Tag is SmartPoster Record \n");
    UARTprintf("    SmartPoster Title:'");
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    for(x=0;x<sNDEFSmartPoster.sTextPayload.ui32TextLength;x++)
    {
        UARTprintf("%c",sNDEFSmartPoster.sTextPayload.pui8Text[x]);
    }
    UARTprintf("'\n");
    UARTprintf("    SmartPoster URI:'");
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    for(x=0;x<sNDEFSmartPoster.sURIPayload.ui32URILength;x++)
    {
        UARTprintf("%c",sNDEFSmartPoster.sURIPayload.puiUTF8String[x]);
    }
    UARTprintf("'\n");
    if(sNDEFSmartPoster.bActionExists)
    {
        switch(sNDEFSmartPoster.sActionPayload.eAction)
        {
            case DO_ACTION:
            {
                UARTprintf("    SmartPoster Action: Do Action (0x%x)\n",
                            sNDEFSmartPoster.sActionPayload.eAction);
                break;
            }
            case SAVE_FOR_LATER:
            {
                UARTprintf("    SmartPoster Action: Save for Later (0x%x)\n",
                            sNDEFSmartPoster.sActionPayload.eAction);
                break;
            }
            case OPEN_FOR_EDITING:
            {
                UARTprintf("    SmartPoster Action: Open for Editing (0x%x)\n",
                            sNDEFSmartPoster.sActionPayload.eAction);
                break;
            }
            default:
            {
                break;
            }
        }
    }
    else
    {
        UARTprintf("    SmartPoster Action: Not Present\n");
    }
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    return;
}

//*****************************************************************************
//
// Print Signiture Record Debug Info
//
//*****************************************************************************
void
DebugSignitureRecord(void)
{
    UARTprintf("    Tag is Signature Record \n");
    UARTprintf("    Signature Records are not fully supported yet.\n");
    #if defined(UART_BUFFERED)
    UARTFlushTx(false);
    #endif
    return;
}
