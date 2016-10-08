//*****************************************************************************
//
// nfc_p2p_demo.c - Simple example for recognising when an NFC card is present,
// reading and writing data to card, and using P2P mode
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
// This is part of revision 2.1.3.156 of the EK-TM4C1294XL Firmware Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "inc/hw_types.h"
#include "driverlib/fpu.h"
#include "driverlib/pin_map.h"
#include "inc/hw_memmap.h"
#include "inc/hw_ints.h"
#include "driverlib/interrupt.h"
#include "driverlib/gpio.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "driverlib/udma.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/timer.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "nfclib/nfc_p2p.h"
#include "nfclib/nfc.h"
#include "nfclib/debug.h"
#include "nfclib/trf79x0.h"
#include "drivers/buttons.h"
#include "drivers/pinout.h"
#include "./trf79x0_hw.h"
#include "./nfc_p2p_demo_debug.h"

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ulLine)
{
    UARTprintf("err: Assert Triggered at line %d in %s\n",ulLine,pcFilename);
}
#endif

//*****************************************************************************
//
// Debug Over UART. Abstracted as Macros to functions in nfc_p2p_demo_debug to
//  make this code easier to understand
//
//*****************************************************************************
#define DEBUG_HEADER(x)                DebugHeader(x)
#define DEBUG_TEXTRECORD(x)            DebugTextRecord(x)
#define DEBUG_URIRECORD(x)             DebugURIRecord(x)
#define DEBUG_SMARTPOSTER(x)           DebugSmartPosterRecord(x)
#define DEBUG_SIGNITURE(x)             DebugSignitureRecord(x)

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>NFC P2P Demo (nfc_p2p_demo)</h1>
//!
//! This example application demonstrates the operation of the Tiva C Series
//! evaluation kit with the TRF7970ABP BoosterPack as a NFC P2P device.
//!
//! The application supports reading and writing Text, URI, and SmartPoster
//! Tags. The application gets a raw message buffer from the TRF79x0 stack,
//! decodes the information to recognized tag types, then re-encodes the data
//! to a buffer to be sent back out. Pressing switch SW1 sends a URI message
//! with a link to the Tiva C series Launchpad website. Pressing switch SW2
//! echo's back the last tag recieved. If no tag has been recieved then this
//! button does nothing. Full debug information is given across the UART0
//! channel to aid in NFC P2P development.
//!
//! This application assumes the TRF7970ABP is connected to the boosterpack 2
//! headers on the development kit. To use the boosterpack 1 headers you will
//! need to toggle the TRF79X0_USE_BOOSTERPACK_2 define in trf79x0_hw.h
//! and recompile the application.
//!
//! For more information on NFC please see the full NFC specification list at
//! http://www.nfc-forum.org/specs/spec_list/ .
//
//*****************************************************************************

//*****************************************************************************
//
// Global to hold clock frequency, write once read many
//
//*****************************************************************************
uint32_t g_ui32SysClk;

//*****************************************************************************
//
// Globals for Timer0A Timeout functions
//
//*****************************************************************************
uint8_t *g_pui8TimeoutPtr;
uint32_t g_ui32TimeoutMsCounter;

//*****************************************************************************
//
// Global Buffer to hold current NFC Tag (set by Decoders, used by Encoders).
// The buffer that holds the last transmitted tag in the stack is highly
// volatile and can change every time a device comes into the field, whether
// there is data or not. This buffer is used as a known good copy of the last
// transmitted tag and is only overwritten when a new tag is transmitted to the
// board.
//
//*****************************************************************************
static uint8_t g_ui8NFCP2PRawTagBuffer[SNEP_MAX_BUFFER];

//*****************************************************************************
//
// Bufferes for Encoding NFC Messages
//
//*****************************************************************************
uint8_t g_pui8MessageBuffer[SNEP_MAX_BUFFER];
uint8_t g_pui8PayloadBuffer[SNEP_MAX_BUFFER];

//*****************************************************************************
//
// NFC NDEF Message Containers. These Structs are used in combination with
// the decode functions to extract data out of a raw NFC data buffer. They are
// also used with the encode functions to recreate the raw data in preperation
// for sending it.
//
//*****************************************************************************
sNDEFMessageData        g_sNDEFMessage;
sNDEFTextRecord         g_sNDEFText;
sNDEFURIRecord          g_sNDEFURI;
sNDEFSmartPosterRecord  g_sNDEFSmartPoster;

//*****************************************************************************
//
// Receive Status Object from Low Level SNEP/NFC Stack
//
//*****************************************************************************
sNFCP2PRxStatus         g_sTRFReceiveStatus;

//*****************************************************************************
//
// Timer0A Initialization function. Use once at startup to setup Timer0A
//
//*****************************************************************************
void
Timer0AInit(void)
{
    //
    // The Timer0 peripheral must be enabled for use.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);

    //
    // The Timer0 peripheral must be enabled for use.
    //
    TimerConfigure(TIMER0_BASE, TIMER_CFG_SPLIT_PAIR | TIMER_CFG_A_PERIODIC);
    //TimerConfigure(TIMER0_BASE, TIMER_CFG_SPLIT_PAIR | TIMER_CFG_A_ONE_SHOT);

    //
    // Configure the Timer0A interrupt for timer timeout.
    //
    TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

    //
    // Enable the Timer0A interrupt on the processor (NVIC).
    //
    IntEnable(INT_TIMER0A);
}

//*****************************************************************************
//
// Timer0A Set function (initialize with value and run)
// \param uint16_t ui16Timeoutms has a maximum value of 65535 mS = 65.5 seconds
// \param uint8_t *ui8TimeouFlag is the flag to set if the time is elapsed.
//
//*****************************************************************************
void
TimerSet(uint16_t ui16Timeoutms, uint8_t *ui8TimeouFlag)
{
    //
    // Check corner case and start timer.
    //
    if(ui16Timeoutms > 0)
    {
        //
        // Set the Timer0A load value to 1ms.
        //
        TimerLoadSet(TIMER0_BASE, TIMER_A,  g_ui32SysClk / 10000);

        //
        // set globals
        //
        g_ui32TimeoutMsCounter = ui16Timeoutms*10;
        g_pui8TimeoutPtr = ui8TimeouFlag;

        //
        // Enable Timer0A Interrupt
        //
        TimerEnable(TIMER0_BASE, TIMER_A);

        //
        // Clear Flag
        //
        *ui8TimeouFlag = 0x00;
    }
    else
        //
        // Corner Case, timeout of 0 means there is nothing to do
        //
        *ui8TimeouFlag = 0x01;
}

//*****************************************************************************
//
// Timer0A Interrupt Handler Routine: counts down global timout in ms, then
// puts a flag up.
//
//*****************************************************************************
void
Timer0AIntHandler(void)
{
    //
    // Clear flags
    //
    TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

    //
    // Subtract off 1ms from global counter, set flag when end is reached
    //
    if(--g_ui32TimeoutMsCounter == 0)
    {
        *g_pui8TimeoutPtr = 0x01;

        //
        // Disable Timer0A Interrupt as it is no longer needed
        //
        TimerDisable(TIMER0_BASE, TIMER_A);
    }
}

//*****************************************************************************
//
// Send the currently encoded tag in the buffer.
// This is a good example of all the fields that must be considered and the
// order to process them in when sending a message.
//
//*****************************************************************************
void
SendData(void)
{
    uint32_t ui32length=0, ui32Type=0, ui32Counter=0, x=0;
    bool bCheck;

    //
    // Determine Tag Type
    //
    for(x=0,ui32Type=0;x<g_sNDEFMessage.ui8TypeLength;x++)
    {
        ui32Type=(ui32Type<<8)+g_sNDEFMessage.pui8Type[x];
    }

    //
    // Handler for different Message Types
    //
    switch(ui32Type)
    {
        case NDEF_TYPE_TEXT :
        {
            //
            // Encode the Record from Struct to Buffer
            //
            bCheck = NFCP2P_NDEFTextRecordEncoder(g_sNDEFText,
                                                    g_pui8PayloadBuffer,
                                                    sizeof(g_pui8PayloadBuffer),
                                                    &ui32Counter);
            break;
        }
        case NDEF_TYPE_URI :
        {
            //
            // Encode the Record from Struct to Buffer
            //
            NFCP2P_NDEFURIRecordEncoder(g_sNDEFURI, g_pui8PayloadBuffer,
                                            sizeof(g_pui8PayloadBuffer),
                                            &ui32Counter);
            break;
        }
        case NDEF_TYPE_SIGNATURE :
        {
            //
            // Do Nothing, Signature Record Types Not Supported
            //

            break;
        }
        case NDEF_TYPE_SMARTPOSTER :
        {
            //
            // Encode the Record from Struct to Buffer
            //
            NFCP2P_NDEFSmartPosterRecordEncoder(g_sNDEFSmartPoster,
                                                g_pui8PayloadBuffer,
                                                sizeof(g_pui8PayloadBuffer),
                                                &ui32Counter);
            break;
        }
        default:
        {
            //
            // The Tag Type is unrecognized. Print Error Message and Exit.
            //
            UARTprintf("Error on Sending Tag. Tag Format Not Recognized.\n");
            return;
            break;
        }
    }

    //
    // Point Payload pointer to encoded payload buffer
    //
    g_sNDEFMessage.pui8PayloadPtr=g_pui8PayloadBuffer;

    //
    // Set Length of Payload
    //
    g_sNDEFMessage.ui32PayloadLength=ui32Counter;

    //
    // Encode Message Header from struct to Buffer
    //  This is used to echo the tag back over NFC
    //
    NFCP2P_NDEFMessageEncoder(g_sNDEFMessage, g_pui8MessageBuffer,
                                sizeof(g_pui8MessageBuffer),&ui32length);

    //
    // Send the NFC data to the stack for processing.
    //
    NFCP2P_sendPacket(g_pui8MessageBuffer,ui32length);

    //
    // Notify User of Message Sent
    //
    UARTprintf("Message Sent.\n");
}

//*****************************************************************************
//
// Send a URI to the DK-TM4C129X webpage to a smartphone / tablet.
// this is a good example of how to create a tag from scratch and send the data.
//
// This example sends a URL to the phone (URL is a type of URI). The standard
// Android response is to prompt the user to open a URL link.
//
//*****************************************************************************
void
SendTIInfo(void)
{
    uint8_t  ui8TIWebpage[]="ti.com/tiva-c-launchpad";
    uint32_t ui32length;
    bool bCheck;

    //
    // Set Header Information
    // This says the message is 1 record long, sent in 1 burst, is short,
    // has no ID field, and is a well known type of record.
    //
    g_sNDEFMessage.sStatusByte.MB=1;     // Message Begin
    g_sNDEFMessage.sStatusByte.ME=1;     // Message End
    g_sNDEFMessage.sStatusByte.CF=0;     // Record is Not Chunked
    g_sNDEFMessage.sStatusByte.SR=1;     // Record is Short Record
    g_sNDEFMessage.sStatusByte.IL=0;     // ID Length =0 (No ID Field Present)
    g_sNDEFMessage.sStatusByte.TNF=TNF_WELLKNOWNTYPE;

    //
    // Set Type to URI ('U')
    // Set Type Lengh to 1 Character
    //
    g_sNDEFMessage.pui8Type[0]='U';     // 'U' is the Type for URI's
    g_sNDEFMessage.ui8TypeLength=1; // TypeLengh is 1 char long ('U')
    //g_sNDEFMessage.ui8IDLength= ; //not needed, IL=0 so no ID is given
    //g_sNDEFMessage.pui8ID=;       //not needed, IL=0 so no ID is given

    //
    // Set URI Record Info
    //
    // prepend the URI with 'http://www.' (see nfc_p2p.h for a full list)
    // set the webpage as the URI string
    // Set the Length of the String
    //
    g_sNDEFURI.eIDCode=http_www;
    g_sNDEFURI.puiUTF8String=ui8TIWebpage;
    g_sNDEFURI.ui32URILength=sizeof(ui8TIWebpage);

    //
    // Encode the URI Record into the Payload Buffer, will return the length
    // of the buffer written in the ui32length variable
    //
    bCheck = NFCP2P_NDEFURIRecordEncoder(g_sNDEFURI,g_pui8PayloadBuffer,
                                       sizeof(g_pui8PayloadBuffer),&ui32length);
    if(!bCheck)
    {
        //
        // NDEFURIRecordEncoder function failed. Alert User
        //
        ASSERT(0);
        UARTprintf("ERR: URI Encoder: Failed in SendTIInfo()\n");
    }

    //
    // Set the Length of the Payload and the Payload Pointer in the
    // Header Struct
    //
    g_sNDEFMessage.ui32PayloadLength=ui32length;
    g_sNDEFMessage.pui8PayloadPtr=g_pui8PayloadBuffer;

    //
    // Encode the Header and the Payload into the Message Buffer
    //
    bCheck = NFCP2P_NDEFMessageEncoder(g_sNDEFMessage,g_pui8MessageBuffer,
                                       sizeof(g_pui8MessageBuffer),&ui32length);
    if(!bCheck)
    {
        //
        // NDEFMessageEncoder function failed. Alert User
        //
        ASSERT(0);
        UARTprintf("ERR: Message Encoder: Failed in SendTIInfo()\n");
    }

    //
    // Send the NFC Message data to the stack for processing.
    //
    NFCP2P_sendPacket(g_pui8MessageBuffer,ui32length);

    //
    // Provide Feedback to User
    //
    UARTprintf("TI URL Sent.\n");

    return;
}

//*****************************************************************************
//
// The main routine
//
//*****************************************************************************
int
main(void)
{
    int TypeID;
    tTRF79x0TRFMode eCurrentTRF79x0Mode = P2P_PASSIVE_TARGET_MODE;
    uint32_t x;
    uint16_t ui16MaxSizeRemaining = 0;
    bool bCheck=STATUS_FAIL;
    uint8_t ui8ButtonDebounced = 0, ui8ButtonDelta = 0, ui8ButtonRaw = 0;
    uint8_t pui8Instructions[]="Instructions:\n "
                "You will need a NFC capable device and a NFC boosterpack for "
                "this demo.\n "
                "To use this demo put the phone or tablet within 2 inches of "
                "the NFC boosterpack.\n "
                "Messages sent to the microcontroller will be displayed on "
                "the terminal.\n "
                "Button SW1 will send a website link to the TI product page "
                "for the board.\n"
                "Button SW2 will echo the last tag sent to the board back to "
                "the phone / tablet.\n";

    //
    // Select NFC Boosterpack Type
    //
    g_eRFDaughterType = RF_DAUGHTER_TRF7970ABP;

    //
    // Run from the PLL at 120 MHz.
    //
    g_ui32SysClk = SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                       SYSCTL_OSC_MAIN |
                                       SYSCTL_USE_PLL |
                                       SYSCTL_CFG_VCO_480), 120000000);

    //
    // Configure the device pins.
    //
    PinoutSet(false, false);

    //
    // Configure the TRF79X0 pins.
    //
    ROM_GPIOPinConfigure(TRF79X0_CLK_CONFIG);
    ROM_GPIOPinConfigure(TRF79X0_TX_CONFIG);
    ROM_GPIOPinConfigure(TRF79X0_RX_CONFIG);

    //
    // Initilize Buttons
    //
    ButtonsInit();

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, g_ui32SysClk);

    //
    // Initialize the TRF79x0 and SSI
    //
    TRF79x0Init();

    //
    // Initialize Timer0A
    //
    Timer0AInit();

    //
    // Enable First Mode
    //
    NFCP2P_init(eCurrentTRF79x0Mode,FREQ_212_KBPS);

    //
    // Enable Interrupts.
    //
    IntMasterEnable();

    //
    // Print a prompt to the console.
    //
    UARTprintf("\n****************************\n");
    UARTprintf("*       NFC P2P Demo       *\n");
    UARTprintf("****************************\n");

    //
    // Print instructions to the console
    //
    UARTprintf((char *)pui8Instructions);

    while(1)
    {
        //
        // NFC-P2P-Initiator-Statemachine
        //
        if(NFCP2P_proccessStateMachine() == NFC_P2P_PROTOCOL_ACTIVATION)
        {
            if(eCurrentTRF79x0Mode == P2P_INITATIOR_MODE)
            {
                eCurrentTRF79x0Mode = P2P_PASSIVE_TARGET_MODE;

                //
                //Toggle LED's
                //
                LEDWrite(CLP_D3,CLP_D3);
                LEDWrite(CLP_D4,0);
            }
            else if(eCurrentTRF79x0Mode == P2P_PASSIVE_TARGET_MODE)
            {
                eCurrentTRF79x0Mode = P2P_INITATIOR_MODE;

                //
                //Toggle LED's
                //
                LEDWrite(CLP_D4,CLP_D4);
                LEDWrite(CLP_D3,0);
            }

            //
            // Initiator switch to Target mode or vice versa
            //
            NFCP2P_init(eCurrentTRF79x0Mode,FREQ_212_KBPS);
        }

        //
        // Read the receive status structure - check if there is a received
        // packet from the Target
        //
        g_sTRFReceiveStatus = NFCP2P_getReceiveState();
        if(g_sTRFReceiveStatus.eDataReceivedStatus != RECEIVED_NO_FRAGMENT)
        {
            //
            //Copy Volatile stack buffer into semi-stable buffer for processing.
            //
            for(x=0;x<g_sTRFReceiveStatus.ui8DataReceivedLength;x++)
            {
                g_ui8NFCP2PRawTagBuffer[x]=g_sTRFReceiveStatus.pui8RxDataPtr[x];

            }

            //
            // Decode Message Header from Buffer to Struct
            //
            bCheck = NFCP2P_NDEFMessageDecoder(&g_sNDEFMessage,
                                            g_ui8NFCP2PRawTagBuffer,
                                            sizeof(g_ui8NFCP2PRawTagBuffer));

            //
            // Check Message Decode Validity
            //
            if(!bCheck)
            {
                //
                // Message Decode Failed, notify user
                //
                UARTprintf("ERR: Message Decode Failed in Main Loop\n");
            }
            if(bCheck)
            {
                //
                // Mesasge Decoded Successfully, Continue Processing.
                //

                //
                // Print Header debug info over UART to Terminal
                //
                DEBUG_HEADER(g_sNDEFMessage);

                //
                // Determine TypeID
                //
                for(x=0,TypeID=0;x<g_sNDEFMessage.ui8TypeLength;x++)
                {
                    TypeID=(TypeID<<8)+g_sNDEFMessage.pui8Type[x];
                }

                //
                // Handler for different Message Types
                //
                switch(TypeID)
                {
                    case NDEF_TYPE_TEXT:
                    {
                        //
                        // Calculate maximum size remaining in buffer
                        // The size remaining = Total size - size used by
                        // header
                        //
                        ui16MaxSizeRemaining =
                            (sizeof(g_ui8NFCP2PRawTagBuffer) -
                            (g_sNDEFMessage.pui8PayloadPtr -
                            &g_ui8NFCP2PRawTagBuffer[0]));

                        //
                        // Decode the Record from Buffer to Struct
                        //
                        NFCP2P_NDEFTextRecordDecoder(&g_sNDEFText,
                                            g_sNDEFMessage.pui8PayloadPtr,
                                            g_sNDEFMessage.ui32PayloadLength);

                        //
                        // Print Record debug info over UART to Terminal
                        //
                        DEBUG_TEXTRECORD(g_sNDEFText);
                        break;
                    }
                    case NDEF_TYPE_URI:
                    {
                        //
                        // Calculate maximum size remaining in buffer
                        // The size remaining = Total size - size used by header
                        //
                        ui16MaxSizeRemaining =
                            (sizeof(g_ui8NFCP2PRawTagBuffer) -
                            (g_sNDEFMessage.pui8PayloadPtr -
                            &g_ui8NFCP2PRawTagBuffer[0]));

                        //
                        // Decode the Record from Buffer to Struct
                        //
                        NFCP2P_NDEFURIRecordDecoder(&g_sNDEFURI,
                                            g_sNDEFMessage.pui8PayloadPtr,
                                            g_sNDEFMessage.ui32PayloadLength);

                        //
                        // Print Record debug info over UART to Terminal
                        //
                        DEBUG_URIRECORD(g_sNDEFURI);
                        break;
                    }
                    case NDEF_TYPE_SIGNATURE:
                    {
                        UARTprintf("Signature Record Not Supported\n");
                        break;
                    }
                    case NDEF_TYPE_SMARTPOSTER:
                    {
                        //
                        // Calculate maximum size remaining in buffer
                        // The size remaining = Total size - size used by header
                        //
                        ui16MaxSizeRemaining =
                            (sizeof(g_ui8NFCP2PRawTagBuffer) -
                            (g_sNDEFMessage.pui8PayloadPtr -
                            &g_ui8NFCP2PRawTagBuffer[0]));

                        //
                        // Decode the Record from Buffer to Struct
                        //
                        NFCP2P_NDEFSmartPosterRecordDecoder(
                                    &g_sNDEFSmartPoster,
                                    g_sNDEFMessage.pui8PayloadPtr,
                                    ui16MaxSizeRemaining,
                                    g_sNDEFMessage.ui32PayloadLength);

                        //
                        // Print Record debug info over UART to Terminal
                        //
                        DEBUG_SMARTPOSTER(g_sNDEFSmartPoster);
                        break;
                    }
                    default:
                    {
                        UARTprintf("    Err: TypeID of Tag Not Recognized: ");
                        for(x=0,TypeID=0;x<g_sNDEFMessage.ui8TypeLength;x++)
                        {
                            UARTprintf("%c",g_sNDEFMessage.pui8Type[x]);
                        }
                        UARTprintf("\n");
                        break;
                    }
                }
            }
        }

        //
        // Get debpimced button state.
        //
        ui8ButtonDebounced = ButtonsPoll(&ui8ButtonDelta, &ui8ButtonRaw);

        //
        // Check Left Button, SW1, send TI information.
        //
        if(BUTTON_PRESSED(LEFT_BUTTON, ui8ButtonDebounced, ui8ButtonDelta))
        {
            SendTIInfo();
        }
        //
        // Check Right Button, SW2, echo current tag in buffer.
        //
        else if(BUTTON_PRESSED(RIGHT_BUTTON, ui8ButtonDebounced,
                                ui8ButtonDelta))
        {
            SendData();
        }
    }
}
