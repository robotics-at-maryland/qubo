//*****************************************************************************
//
// shamd5_hmac_example.c - Simple SHA/MD5 HMAC Example
//
// Copyright (c) 2015-2016 Texas Instruments Incorporated.  All rights reserved.
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
#include "driverlib/shamd5.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "utils/cmdline.h"
#include "utils/ustdlib.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>SHA/MD5 HMAC Demo (shamd5_hmac_example)</h1>
//!
//! This application demonstrates the use of HMAC operation with the available
//! algorithms of the SHA/MD5 module like MD5, SHA1, SHA224 and SHA256.
//!
//! This application uses a command-line based interface through a virtual COM
//! port on UART 0, with the settings 115,200-8-N-1.
//!
//! Using the command prompt, user can configure the SHA/MD5 module to select
//! an algorithm to perform HMAC operation.  User can also enter key and data
//! values during runtime.  Type "help" on the terminal, once the prompt is
//! displayed, for details of these configuration.
//!
//! The following web site has been used to validate the HMAC output for these
//! algorithms: http://www.freeformatter.com/hmac-generator.html
//!
//! Please note that uDMA is not used in this example.
//
//*****************************************************************************

//*****************************************************************************
//
// Configuration defines.
//
//*****************************************************************************
#define CMD_BUF_SIZE            64
#define HMAC_KEY_DATA_SIZE      256

//*****************************************************************************
//
// Command Line Buffer
//
//*****************************************************************************
static char g_pcCmdBuf[CMD_BUF_SIZE];

//*****************************************************************************
//
// Key, Plain Text Data Buffers and Size Variable
//
//*****************************************************************************
static char g_pcHMACKey[HMAC_KEY_DATA_SIZE];
static char g_pcHMACData[HMAC_KEY_DATA_SIZE];
uint32_t g_ui32HMACSizeInBytes;

//*****************************************************************************
//
// This function implements the "reset" command.  It resets the CCM0 Module and
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
    // Wait for the peipheral to be ready.
    //
    while(!MAP_SysCtlPeripheralReady(SYSCTL_PERIPH_CCM0))
    {
    }

    //
    // Reset the SHA/MD5 module.
    //
    MAP_SHAMD5Reset(SHAMD5_BASE);

    return 0;
}

//*****************************************************************************
//
// This function implements the "algo" command.  It selects the HMAC algorithm
// to be used for hashing and selects the size of the HASH output.
//
//*****************************************************************************
int
Cmd_algo(int argc, char *argv[])
{
    uint32_t ui32Algo;

    //
    // Check if correct argument is passed.
    //
    if(argc != 2)
    {
        //
        // No - Print error message and exit.
        //
        UARTprintf("\nInvalid arguments passed.  This command takes only one "
                   "argument.\n");
        return 0;
    }

    //
    // Check if the argument passed has correct value.  If yes, then process
    // the argument.
    //
    if(ustrcmp("md5",argv[1]) == 0 )
    {
        ui32Algo = SHAMD5_ALGO_HMAC_MD5;
        g_ui32HMACSizeInBytes = 16;
    }
    else if(ustrcmp("sha1",argv[1]) == 0 )
    {
        ui32Algo = SHAMD5_ALGO_HMAC_SHA1;
        g_ui32HMACSizeInBytes = 20;
    }
    else if(ustrcmp("sha224",argv[1]) == 0 )
    {
        ui32Algo = SHAMD5_ALGO_HMAC_SHA224;
        g_ui32HMACSizeInBytes = 28;
    }
    else if(ustrcmp("sha256",argv[1]) == 0 )
    {
        ui32Algo = SHAMD5_ALGO_HMAC_SHA256;
        g_ui32HMACSizeInBytes = 32;
    }
    else
    {
        //
        // Argument passed is not supported.  Inform user and exit.
        //
        UARTprintf("\nHMAC algorithm passed is not supported.\n");
        UARTprintf("Syntax: algo <HMAC>\n"
                   "     <HMAC> takes one of md5, sha1, sha224 or sha256");

        return 0;
    }

    //
    // Configure the SHA/MD5 module for the algorithm requested.
    //
    MAP_SHAMD5ConfigSet(SHAMD5_BASE, ui32Algo);

    return 0;
}

//*****************************************************************************
//
// This function implements the "key" command.  It allows the user to provide
// key for HMAC operation.
//
//*****************************************************************************
int
Cmd_key(int argc, char *argv[])
{
    uint32_t ui32Index;

    //
    // Check if at least one key value is passed.
    //
    if(argc < 2)
    {
        //
        // No - Print error message and exit.
        //
        UARTprintf("\nNo argument passed.  This command requires atleast one "
                   "argument as key.\n");
        UARTprintf("The key can contain spaces.\n");
        return 0;
    }

    //
    // Copy the first value after the command into the final string.  All
    // subsequent values are concatenated to the final string.
    //
    ui32Index = 1;
    ustrncpy(g_pcHMACKey, argv[ui32Index], sizeof(g_pcHMACKey));
    ui32Index++;
    while(ui32Index < argc)
    {
        strcat(g_pcHMACKey, " ");
        strcat(g_pcHMACKey, argv[ui32Index]);
        ui32Index++;
    }

    return 0;
}

//*****************************************************************************
//
// This function implements the "data" command.  It allows the user to provide
// data for HMAC operation.
//
//*****************************************************************************
int
Cmd_data(int argc, char *argv[])
{
    uint32_t ui32Index;

    //
    // Check if at least one key value is passed.
    //
    if(argc < 2)
    {
        //
        // No - Print error message and exit.
        //
        UARTprintf("\nNo argument passed.  This command requires atleast one "
                   "argument as data.\n");
        UARTprintf("Data can contain spaces.\n");
        return 0;
    }

    //
    // Copy the first value after the command into the final string.  All
    // subsequent values are concatenated to the final string.
    //
    ui32Index = 1;
    ustrncpy(g_pcHMACData, argv[ui32Index], sizeof(g_pcHMACData));
    ui32Index++;
    while(ui32Index < argc)
    {
        strcat(g_pcHMACData," ");
        strcat(g_pcHMACData, argv[ui32Index]);
        ui32Index++;
    }

    return 0;
}

//*****************************************************************************
//
// This function implements the "hmac" command, which starts the process to
// calculate the HMAC value of the provided data with the provided key.
//
//*****************************************************************************
int
Cmd_hmac(int argc, char *argv[])
{
    uint32_t ui32Index;
    uint32_t pui32HashResult[32];
    uint8_t  *pui8Ptr;

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
    // Copy the Key to SHA/MD5 module.
    //
    MAP_SHAMD5HMACKeySet(SHAMD5_BASE, (uint32_t *) g_pcHMACKey);

    //
    // Copy the data to SHA/MD5 module and begin the HMAC generation.
    //
    MAP_SHAMD5HMACProcess(SHAMD5_BASE, (uint32_t *) g_pcHMACData,
                      strlen(g_pcHMACData), pui32HashResult);

    //
    // Copy the HMAC output from a 32-bit pointer to an 8-bit pointer, to help
    // in displaying the output.
    //
    pui8Ptr = (uint8_t *)&pui32HashResult[0];

    //
    // Print the final HMAC output.
    //
    UARTprintf("\nHASH OUTPUT\n");
    for(ui32Index = 0; ui32Index < g_ui32HMACSizeInBytes;
        ui32Index++, pui8Ptr++)
    {
        UARTprintf("%02x", *pui8Ptr);
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
        UARTprintf("%6s: %s\n", psEntry->pcCmd, psEntry->pcHelp);

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
    { "reset",  Cmd_reset,  " Resets the Crypto Modules" },
    { "algo",   Cmd_algo,   " Selects HMAC algorithm\n\t     Syntax:"
                            " algo <HMAC>\n\t     <HMAC> takes one of md5,"
                            " sha1, sha224 or sha256"},
    { "key",    Cmd_key,    " Enter Key for HMAC operation\n\t     Syntax:"
                            " key <KEY>\n\t     <KEY> can take spaces" },
    { "data",   Cmd_data,   " Enter Key for HMAC operation\n\t     Syntax:"
                            " data <DATA>\n\t     <DATA> can take spaces" },
    { "hmac",   Cmd_hmac,   " Output HMAC based on:\n\t "
                            "   * algo - set HMAC algorithm\n\t "
                            "   * key - enter key\n\t "
                            "   * data - enter data" },
    { 0, 0, 0 }
};

//*****************************************************************************
//
// This is the main function to implement the SHA/MD5 HMAC operation.
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
    // Disable clock to CCM0 and reset the module from system control and then
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
    // Reset the SHA/MD5 module.
    //
    MAP_SHAMD5Reset(SHAMD5_BASE);

    //
    // Clear the screen and print a welcome message.
    //
    UARTprintf("\033[2J\033[1;1HHash Message Authentication Code Example\n\n");
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
