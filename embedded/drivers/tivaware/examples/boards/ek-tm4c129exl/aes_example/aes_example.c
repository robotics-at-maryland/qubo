//*****************************************************************************
//
// aes_example.c - Simple AES Encryption-Decryption Example
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
// This is part of revision 2.1.1.71 of the EK-TM4C129EXL Firmware Package.
//
//*****************************************************************************
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "inc/hw_gpio.h"
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_sysctl.h"
#include "inc/hw_types.h"
#include "inc/hw_uart.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/aes.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "utils/cmdline.h"
#include "utils/ustdlib.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>AES Encryption and Decryption Demo (aes_example)</h1>
//!
//! This application demonstrates encryption and decryption for the available
//! modes of the AES module.
//!
//! This application uses a command-line based interface through a virtual COM
//! port on UART 0, with the settings 115,200-8-N-1.
//!
//! Using the command prompt, user can configure the AES module to select the
//! mode, key-size, and direction (encryption/decryption) during runtime.  User
//! can also enter key, data and IV values during runtime.  Type "help" on the
//! terminal, once the prompt is displayed, for details of these configuration.
//!
//! The examples from NIST specification at the following link have been used
//! to validate the AES outptut.
//! http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf
//!
//! Please note that uDMA is not used in this example.
//
//*****************************************************************************

//*****************************************************************************
//
// Configuration defines.
//
//*****************************************************************************
#define CMD_BUF_SIZE            256
#define AES_KEY_DATA_SIZE       256
#define AES_IV_DATA_SIZE        16
#define AES_DATA_SIZE           256

//*****************************************************************************
//
// Command Line Buffer
//
//*****************************************************************************
static char g_pcCmdBuf[CMD_BUF_SIZE];

//*****************************************************************************
//
// Key, Plain Text Data, Initialization Vector Buffers and Size Variable
//
//*****************************************************************************
static char g_pcAESKey[AES_KEY_DATA_SIZE];
static char g_pcAESEncryptDataIn[AES_DATA_SIZE];
static uint8_t g_pui8AESIVData[AES_IV_DATA_SIZE];
static uint8_t g_pui8AESEncryptDataOut[AES_DATA_SIZE];
static uint32_t g_ui32KeySize;
volatile bool g_bIVRequired;
volatile bool g_bProcessDirection;
volatile uint32_t g_ui32StringLength;

//*****************************************************************************
//
// This function converts the hex characters from a source buffer into integers
// and places them into the destination buffer.
//
// Each character of the source buffer represents a nibble of the hexadecimal
// number.  So two characters from the source buffer will be combined to form a
// single byte of the destination buffer.
//
//*****************************************************************************
int
CharToHex(char *pcSrc, uint32_t ui32Len, uint8_t *pcDst)
{
    uint32_t ui32OutCode, ui32CharAvailable, ui32CharToProcess, ui32Index;
    char pcTemp[9];

    //
    // Check if the soruce and destination buffers are valid.  If so return
    // error.
    //
    if((pcSrc == NULL) || (pcDst == NULL))
    {
        return 0;
    }

    //
    // Get the number of characters available in the source buffer for
    // conversion.
    //
    ui32CharAvailable = ui32Len;

    //
    // Convert all the hex characters in the source buffer into integers.
    //
    while(ui32CharAvailable)
    {
        //
        // Check if 8 or more characters are available in the buffer.
        //
        if(ui32CharAvailable > 7)
        {
            //
            // Yes - then process 8 characters at a time since "ustrtoul"
            // function can convert only upto 8 characters in one go.
            //
            ui32CharToProcess = 8;
        }
        else
        {
            //
            // No - then process the remaining characters.
            //
            ui32CharToProcess = ui32CharAvailable;
        }

        //
        // Copy the characters, to process, into a temp buffer and append end
        // of string.  Copy must be done such that pcSrc[6:7] goes to
        // pcTemp[0:1].  This is done so that copying the output of "ustrtoul"
        // function into destination bufer becomes easy.
        //
        for(ui32Index = 0; ui32Index < ui32CharToProcess; ui32Index += 2)
        {
            pcTemp[ui32Index + 1] = pcSrc[(ui32CharToProcess - 1) - ui32Index];
            pcTemp[ui32Index] = pcSrc[(ui32CharToProcess - 1) -
                                      (ui32Index + 1)];
        }
        pcTemp[ui32CharToProcess] = '\0';

        //
        // Update necessary resources for next iteration.
        //
        pcSrc += ui32CharToProcess;
        ui32CharAvailable -= ui32CharToProcess;

        //
        // Convert the hex characters to integer.
        //
        ui32OutCode =  ustrtoul(pcTemp, 0, 16);

        //
        // Copy the hex digits into the destination buffer.
        //
        while(ui32OutCode)
        {
            *pcDst = (ui32OutCode & 0xFF);
            pcDst++;
            ui32OutCode = (ui32OutCode >> 8);
        }
    }

    //
    // Append end of sting to the destination buffer.
    //
    pcDst = '\0';

    //
    // Return Success.
    //
    return 1;
}

//*****************************************************************************
//
// This function implements the "reset" command.  It resets the CCM0 module and
// enables the clock.
//
//*****************************************************************************
int
Cmd_reset(int argc, char *argv[])
{
    //
    // Disable clock to CCM0 and reset the module from system control and then
    // enable the clock.
    //
    MAP_SysCtlPeripheralDisable(SYSCTL_PERIPH_CCM0);
    MAP_SysCtlPeripheralReset(SYSCTL_PERIPH_CCM0);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_CCM0);

    //
    // Wait for the peipheral to be ready
    //
    while(!MAP_SysCtlPeripheralReady(SYSCTL_PERIPH_CCM0))
    {
    }

    //
    // Reset the AES module.
    //
    MAP_AESReset(AES_BASE);

    //
    // Reset the variable for IV and direction of processing.
    //
    g_bIVRequired = false;
    g_bProcessDirection = false;

    return 0;
}

//*****************************************************************************
//
// This function prints the help instructions for the command "algo".
//
//*****************************************************************************
void
PrintHelpCmdAlgo(void)
{
    UARTprintf("Syntax:\n"
               "  algo <Mode> <Key_Size> <Direction> (IV)\n"
               "    <Mode> - Feedback operating modes for the AES Block\n"
               "      Takes only \"ecb\", \"cbc\" or \"cfb\"\n"
               "    <Key_Size> - Key size in bits\n"
               "      Takes only \"128\", \"192\" or \"256\"\n"
               "    <Direction> - Encryption or Decryption\n"
               "      Takes 1 to \"encrypt\"\n"
               "            0 to \"decrypt\"\n"
               "    (IV) - Optional Initialization Vector - in hex format.\n"
               "      CBC and CFB modes require IV, else default value used.\n"
               "Example - algo cbc 128 1"
               " 000102030405060708090A0B0C0D0E0F\n");
}

//*****************************************************************************
//
// This function implements the "algo" command.  It selects the AES algorithm
// to be used for encryption.
//
//*****************************************************************************
int
Cmd_algo(int argc, char *argv[])
{
    bool bIVRequired;
    uint32_t ui32StrLen, ui32Mode, ui32KeySize, ui32Direction;

    //
    // Check if correct arguments are passed.
    //
    if((argc < 4) || (argc > 5))
    {
        //
        // Error.  Print help message and return.
        //
        PrintHelpCmdAlgo();
        return 0;
    }

    //
    // Validate and process the value passed to <Mode> argument.
    //
    if((ustrncmp("ecb", argv[1], 3) == 0))
    {
        ui32Mode = AES_CFG_MODE_ECB;
        bIVRequired = false;
    }
    else if((ustrncmp("cbc", argv[1], 3) == 0))
    {
        ui32Mode = AES_CFG_MODE_CBC;
        bIVRequired = true;
    }
    else if((ustrncmp("cfb", argv[1], 3) == 0))
    {
        ui32Mode = AES_CFG_MODE_CFB;
        bIVRequired = true;
    }
    else
    {
        //
        // Error - Wrong value passed to <Mode> argument.  Print help message
        // and return.
        //
        UARTprintf("\"Mode\" parameter is invalid.  Must be \"ecb\", \"cbc\" "
                   "or \"cfb\"\n");
        PrintHelpCmdAlgo();
        return 0;
    }

    //
    // Validate and process the value passed to the optional (IV) argument.
    //
    if(bIVRequired == false)
    {
        //
        // For modes that don't need IV, check if IV is passed.
        //
        if(argc > 4)
        {
            //
            // Yes - This is an Error.  Print help message and return.
            //
            UARTprintf("Extra paramter passed.  Note that feedback mode "
                       "\"ecb\" does not take an IV.\n");
            PrintHelpCmdAlgo();
            return 0;
        }
    }
    else
    {
        //
        // For modes that require IV, check if IV is passed.
        //
        if(argc < 5)
        {
            //
            // No - This is an Error.  Print help message and return.
            //
            UARTprintf("Paramter missing.  Note that feedback modes \"cbc\" "
                       "and \"cfb\" nedd an IV value.\n");
            PrintHelpCmdAlgo();
            return 0;
        }

        //
        // An IV value is passed.  Check if the value is valid.
        //
        ui32StrLen = ustrlen(argv[4]);
        if(ui32StrLen != 32)
        {
            //
            // Invalid value.  Print help message and return.
            //
            UARTprintf("IV value must be 16 HEX characters long.\n");
            PrintHelpCmdAlgo();
            return 0;
        }

        //
        // Convert the IV from string to HEX.
        //
        if(!(CharToHex(argv[4], ui32StrLen, &g_pui8AESIVData[0])))
        {
            //
            // Invalid value.  Print help message and return.
            //
            UARTprintf("IV value has non-hex characters.\n");
            PrintHelpCmdAlgo();
            return 0;
        }
    }

    //
    // Validate and process the value passed to <Key_Size> argument.
    //
    ui32KeySize = ustrtoul(argv[2], 0, 10);
    if((ui32KeySize != 128) && (ui32KeySize != 192) && (ui32KeySize != 256))
    {
        //
        // Error - Wrong value passed to <Key_Size> argument.  Print help
        // message and return.
        //
        UARTprintf("\"Key_Size\" parameter is invalid.  Must be \"128\", "
                   "\"192\" or \"256\"\n");
        PrintHelpCmdAlgo();
        return 0;
    }

    //
    // Copy the <Key_Size> argument value into a variable.
    //
    ui32KeySize = (ui32KeySize == 128) ? AES_CFG_KEY_SIZE_128BIT :
                  (ui32KeySize == 192) ? AES_CFG_KEY_SIZE_192BIT :
                  AES_CFG_KEY_SIZE_256BIT;

    //
    // Validate and process the value passed to <Direction> argument.
    //
    ui32Direction = ustrtoul(argv[3], 0, 10);
    if((ui32Direction != 0) && (ui32Direction != 1))
    {
        //
        // Error - Wrong value passed to <Direction> argument.  Print help
        // message and return.
        //
        UARTprintf("\"Direction\" parameter is invalid. Must be \"0\" or "
                   "\"1\".\n");
        PrintHelpCmdAlgo();
        return 0;
    }

    //
    // Copy the <Direction> argument value into a variable.
    //
    ui32Direction = (ui32Direction) ? AES_CFG_DIR_ENCRYPT :
                                      AES_CFG_DIR_DECRYPT;

    //
    // Set-up the AES module with the necessary parameters.
    //
    MAP_AESConfigSet(AES_BASE, (ui32KeySize | ui32Direction | ui32Mode));

    //
    // Update global resources to allow use by the command "process".
    //
    g_ui32KeySize = ui32KeySize;
    g_bIVRequired = bIVRequired;
    g_bProcessDirection = (ui32Direction == AES_CFG_DIR_ENCRYPT) ? false :
                          true;

    return 0;
}

//*****************************************************************************
//
// This function prints the help instructions for the command "key".
//
//*****************************************************************************
void
PrintHelpCmdKey(void)
{
    UARTprintf("Syntax:\n"
               "  key (hex) <KEY>\n"
               "    <KEY> is required\n"
               "      Should be a string of characters or hexadecimals\n"
               "    (hex) is optional\n"
               "      If used, <KEY> must be in hex format\n"
               "      If not used, <KEY> is interpreted as plain text\n");
}

//*****************************************************************************
//
// This function implements the "key" command.  It allows the user to input a
// key for generating the AES encryption
//
//*****************************************************************************
int
Cmd_key(int argc, char *argv[])
{
    uint32_t ui32Index, ui32StrLen;

    //
    // Check if correct arguments are sent.
    //
    if(argc < 2)
    {
        //
        // Error.  Print help message and return.
        //
        PrintHelpCmdKey();
        return 0;
    }

    //
    // Is the optional (hex) parameter entered?
    //
    if(ustrcmp("hex", argv[1]) == 0 )
    {
        //
        // Yes - Check if only one paramter follows (hex).
        //
        if(argc > 3)
        {
            //
            // Error.  Print help message and return.
            //
            UARTprintf("\nInvalid parameters used!\n");
            PrintHelpCmdKey();
            return 0;
        }

        //
        // Does the KEY, entered, have even number of characters?
        //
        ui32StrLen = ustrlen(argv[2]);
        if((ui32StrLen % 2) != 0)
        {
            //
            // No - Error!  Print error message and return.
            //
            UARTprintf("\nODD number of characters entered for KEY.  KEY "
                        "should contain EVEN number of characters.\n");
            return 0;
        }

        //
        // Now convert input HEX stream into a number.
        //
        if(!(CharToHex(argv[2], ui32StrLen, (uint8_t *)g_pcAESKey)))
        {
            UARTprintf("\nKey has non-hex characters.\n");
            return 0;
        }
    }
    else
    {
        //
        // Since the optional parameter (hex) was not entered, assume that user
        // has entered KEY in plain text.  Copy the first value, after the
        // command, into the final string.  Concatenate all subsequent values
        // to the final string.
        //
        ui32Index = 1;
        ustrncpy(g_pcAESKey, argv[ui32Index], sizeof(g_pcAESKey));
        ui32Index++;
        while(ui32Index < argc)
        {
            strcat(g_pcAESKey," ");
            strcat(g_pcAESKey, argv[ui32Index]);
            ui32Index++;
        }
    }
    return 0;
}

//*****************************************************************************
//
// This function prints the help instructions for the command "data".
//
//*****************************************************************************
void
PrintHelpCmdData(void)
{
    UARTprintf("Syntax:\n"
               "  data (hex) <DATA>\n"
               "    <DATA> is required\n"
               "      Should be a string of characters or hexadecimals\n"
               "    (hex) is optional\n"
               "      If used, <DATA> must be in hex format\n"
               "      If not used, <DATA> is interpreted as plain text\n");
}

//*****************************************************************************
//
// This function implements the "data" command.  It allows the user to enter
// the data to be encrypted or decrypted.
//
//*****************************************************************************
int
Cmd_data(int argc, char *argv[])
{
    uint32_t ui32Index, ui32StrLen;

    //
    // Check if correct arguments are sent.
    //
    if(argc < 2)
    {
        //
        // Error.  Print help message and return.
        //
        PrintHelpCmdData();
        return 0;
    }

    //
    // Null the string before reading any data
    //
    for(ui32Index = 0 ; ui32Index < AES_DATA_SIZE ; ui32Index++)
    {
        g_pcAESEncryptDataIn[ui32Index] = 0x0;
    }

    //
    // Is the optional (hex) parameter entered?
    //
    if(ustrcmp("hex",argv[1]) == 0 )
    {
        //
        // Yes - Check if only one paramter follows (hex).
        //
        if(argc > 3)
        {
            //
            // Error.  Print help message and return.
            //
            UARTprintf("\nInvalid parameters used!\n");
            PrintHelpCmdData();
            return 0;
        }

        //
        // Does the DATA, entered, have even number of characters?
        //
        ui32StrLen = ustrlen(argv[2]);
        if((ui32StrLen % 2) != 0)
        {
            //
            // No - Error!  Print error message and return.
            //
            UARTprintf("ODD number of characters entered for DATA.  DATA "
                       "should contain EVEN number of characters.\n");
            return 0;
        }

        //
        // Get the hex string length
        //
        g_ui32StringLength = ui32StrLen / 2;

        //
        // Now convert input HEX stream to a number.
        //
        if(!(CharToHex(argv[2], ui32StrLen, (uint8_t *)g_pcAESEncryptDataIn)))
        {
            UARTprintf("\nData has non-hex characters.\n");
            return 0;
        }
    }
    else
    {
        //
        // Clear the hex string length
        //
        g_ui32StringLength = 0;

        //
        // Since the optional parameter (hex) was not entered, assume that user
        // has entered DATA in plain text.  Copy the first value, after the
        // command, into the final string.  Concatenate all subsequent values
        // to the final string.
        //
        ui32Index = 1;
        ustrncpy(g_pcAESEncryptDataIn, argv[ui32Index],
                 sizeof(g_pcAESEncryptDataIn));
        ui32Index++;
        while(ui32Index < argc)
        {
            strcat(g_pcAESEncryptDataIn," ");
            strcat(g_pcAESEncryptDataIn, argv[ui32Index]);
            ui32Index++;
        }
    }

    return 0;
}

//*****************************************************************************
//
// This function implements the "process" command.  This starts the process to
// encrypt or decrypt the data.
//
//*****************************************************************************
int
Cmd_process(int argc, char *argv[])
{
    uint32_t ui32Index, ui32StringLength, ui32DataBlock;
    int32_t i32DataLine;
    uint8_t *pui8Ptr;

    //
    // Check if correct parameters were passed.
    //
    if(argc != 1)
    {
        //
        // No - Print error message and exit.
        //
        UARTprintf("\nInvalid arguments passed.  This command does not accept "
                   "any arguments.\n");
        return 0;
    }

    //
    // Check if initialization vector is required.
    //
    if(g_bIVRequired)
    {
        //
        // Yes - Write the initial value.
        //
        MAP_AESIVSet(AES_BASE, (uint32_t *) g_pui8AESIVData);
    }

    //
    // Copy the Key to AES module.
    //
    MAP_AESKey1Set(AES_BASE, (uint32_t *) g_pcAESKey, g_ui32KeySize);

    //
    // Get the length of the data.
    //
    if(g_ui32StringLength == 0)
    {
        ui32StringLength = ustrlen(g_pcAESEncryptDataIn);
    }
    else
    {
        ui32StringLength = g_ui32StringLength;
    }

    //
    // Get the block size of Encrypt/Decrypt data to help in displaying the
    // output.
    //
    ui32DataBlock = (((ui32StringLength - 1) / 16) + 1) * 16;

    //
    // Copy the data to AES module and begin the encryption/decryption.
    //
    MAP_AESDataProcess(AES_BASE, (uint32_t *)g_pcAESEncryptDataIn,
                   (uint32_t *)g_pui8AESEncryptDataOut, ui32StringLength);

    //
    // Copy the encrypted/decrypted output from a 32-bit pointer to an 8-bit
    // pointer, to help in displaying the output.
    //
    pui8Ptr = g_pui8AESEncryptDataOut;

    //
    // Initialize the line display.
    //
    i32DataLine = -1;

    //
    // Print the final encrypted/decrypted output.
    //
    if(!g_bProcessDirection)
    {
        UARTprintf("\nENCRYPTED OUTPUT\n");
    }
    else
    {
        UARTprintf("\nDEECRYPTED OUTPUT\n");
    }
    for(ui32Index = 0 ; ui32Index < ui32DataBlock ; ui32Index++, pui8Ptr++)
    {
        if(i32DataLine == (ui32Index / 16))
        {
            UARTprintf("%02x ", *pui8Ptr);
        }
        else
        {
            i32DataLine = (ui32Index / 16);
            UARTprintf("\n%07x0 : %02x ", i32DataLine, *pui8Ptr);
        }
    }

    //
    // Exit.
    //
    return 0;
}

//*****************************************************************************
//
// This function implements the "help" command.  It prints a simple list of the
// available commands with a brief description.
//
//*****************************************************************************
int
Cmd_help(int argc, char *argv[])
{
    tCmdLineEntry *psEntry;

    //
    // Print some header text.
    //
    UARTprintf("\nAvailable commands");
    UARTprintf("\n------------------\n");

    //
    // Point at the beginning of the command table.
    //
    psEntry = &g_psCmdTable[0];

    //
    // Enter a loop to read each entry from the command table.  The end of the
    // table has been reached when the command name is NULL.
    //
    while(psEntry->pcCmd)
    {
        //
        // Print the command name and the brief description.
        //
        UARTprintf("%7s: %s\n", psEntry->pcCmd, psEntry->pcHelp);

        //
        // Advance to the next entry in the table.
        //
        psEntry++;
    }

    //
    // Return success.
    //
    return(0);
}

//*****************************************************************************
//
// This function sets up UART0 to be used for a console to display information
// as the example is running.
//
//*****************************************************************************
void
InitConsole(void)
{
    //
    // Enable GPIO port A which is used for UART0 pins.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    //
    // Configure the pin muxing for UART0 functions on port A0 and A1.
    //
    MAP_GPIOPinConfigure(GPIO_PA0_U0RX);
    MAP_GPIOPinConfigure(GPIO_PA1_U0TX);

    //
    // Enable UART0 so that we can configure the clock.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Use the internal 16MHz oscillator as the UART clock source.
    //
    MAP_UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    //
    // Select the alternate (UART) function for these pins.
    //
    MAP_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, 16000000);
}

//*****************************************************************************
//
// This is the table that holds the command names, implementing functions, and
// brief description.
//
//*****************************************************************************
tCmdLineEntry g_psCmdTable[] =
{
    { "help",   Cmd_help,   " Display list of commands" },
    { "?",      Cmd_help,   " Display list of commands" },
    { "h",      Cmd_help,   " Display list of commands" },
    { "reset",  Cmd_reset,  " Reset the Crypto Modules" },
    { "algo",   Cmd_algo,   " Select AES algorithm, key length,"
                            " encrypt/decrypt and IV\n\t     Syntax:"
                            "algo <Mode> <Key_size> <Direction> <IV>\n\t     "
                            " For more help enter \"algo\" at prompt" },
    { "key",    Cmd_key,    " Enter Key for Encryption\n\t     Syntax:"
                            "key (hex) <KEY>\n\t      For more help enter"
                            " \"key\" at prompt" },
    { "data",   Cmd_data,   " Enter Data for Encryption\n\t     Syntax:"
                            "data (hex) <DATA>\n\t      For more help enter"
                            " \"data\" at prompt" },
    { "process",Cmd_process," Output encrypted/decrypted data based on:\n\t "
                            "   * algo - set AES algorithm\n\t "
                            "   * key - enter key\n\t "
                            "   * data - enter data to encrypt or decrypt" },
    { 0, 0, 0 }
};

//*****************************************************************************
//
// This is the main function to implement the AES Encrytion and Decryption.
//
//*****************************************************************************
int
main(void)
{
    int iStatus;

    //
    // Enable and initialize the UART0 Console.
    //
    InitConsole();

    //
    // Disable clock to CCM0 and reset the module from  system control and then
    // enable the clock.
    //
    MAP_SysCtlPeripheralDisable(SYSCTL_PERIPH_CCM0);
    MAP_SysCtlPeripheralReset(SYSCTL_PERIPH_CCM0);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_CCM0);

    //
    // Wait for the peripheral to be ready.
    //
    while(!MAP_SysCtlPeripheralReady(SYSCTL_PERIPH_CCM0))
    {
    }

    //
    // Reset AES Module.
    //
    MAP_AESReset(AES_BASE);

    //
    // Reset the variable for IV and direction of processing.
    //
    g_bIVRequired = false;
    g_bProcessDirection = false;

    //
    // Clear the screen and print a welcome message.
    //
    UARTprintf("\033[2J\033[1;1HAES Encryption/Decryption Example\n\n");
    UARTprintf("Type help for Options.\n\n");

    //
    // Enter an infinite loop for reading and processing commands from the
    // user.
    //
    while(1)
    {
        //
        // Begin command prompt.
        //
        UARTprintf("\n> ");

        //
        // Get a line of text from the user.
        //
        UARTgets(g_pcCmdBuf, sizeof(g_pcCmdBuf));

        //
        // Pass the line from the user to the command processor.  It will be
        // parsed and valid commands executed.
        //
        iStatus = CmdLineProcess(g_pcCmdBuf);

        //
        // Handle the case of bad command.
        //
        if(iStatus == CMDLINE_BAD_CMD)
        {
            UARTprintf("Bad command!\n");
        }

        //
        // Handle the case of too many arguments.
        //
        else if(iStatus == CMDLINE_TOO_MANY_ARGS)
        {
            UARTprintf("Too many arguments for command processor!\n");
        }
    }
}
