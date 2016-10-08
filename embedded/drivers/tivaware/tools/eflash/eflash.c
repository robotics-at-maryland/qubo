//*****************************************************************************
//
// eflash.c - This file holds the main routine for downloading an image to a
//            Tiva device via Ethernet.
//
// Copyright (c) 2009-2016 Texas Instruments Incorporated.  All rights reserved.
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

#include <stdbool.h>
#include <stdint.h>
#include <winsock2.h>
#include <errno.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ctype.h>
#include <string.h>
#include <signal.h>
#include "eflash.h"
#include "bootp_server.h"

//*****************************************************************************
//
// The version of the application.
//
//*****************************************************************************
const char *g_pcApplicationVersion = "2.1.3.156";

//*****************************************************************************
//
// Program strings used in various display routines.
//
//*****************************************************************************
static const char g_pcProgramName[] =
    "EFLASH Ethernet Boot Loader Download Utility";

static const char g_pcProgramCopyright[] =
    "Copyright (c) 2009-2016 Texas Instruments Incorporated.  All rights reserved.";

static const char g_pcProgramHelp[] =
"usage: eflash [options] file\n"
"\n"
"Download a file to a remote device, using the Ethernet Boot Loader.\n"
"The file should be a binary image, and the IP and MAC address of the\n"
"target device must be specified.\n"
"Example: eflash -i 169.254.19.63 --mac=00.1a.b6.00.12.04 enet_lwip.bin\n"
"\n"
"Required options:\n"
"  -i addr, --ip=addr     IP address of remote device to program,\n"
"                         in dotted-decimal notation\n"
"                         (e.g. 169.254.19.63)\n"
"  -m addr, --mac=addr    MAC address of remote device to program,\n"
"                         specified as a series of hexadecimal numbers\n"
"                         delimited with '-', ':', or '.'.\n"
"                         (e.g. 00.1a.b6.00.12.04)\n"
"  file                   binary file to be downloaded to the remote device.\n"
"                         (e.g. enet_lwip.bin)\n"
"\n"
"Output control:\n"
"      --quiet, --silent  suppress all normal output\n"
"      --verbose          display additional status information\n"
"      --debug            display additional diagnostic information\n"
"\n"
"Miscellaneous:\n"
"      --version          display program version information, then exit\n"
"      --help             display this help text, then exit\n"
"\n"
"Support Information:\n"
"Report any bugs to <support_lmi@ti.com>\n";

//*****************************************************************************
//
// Command line option to set "printf" output display level.
//
//*****************************************************************************
int32_t g_i32OptVerbose = 1;

//*****************************************************************************
//
// MAC address of target (remote) device.
//
//*****************************************************************************
uint8_t g_pui8RemoteMAC[6] = {0, 0, 0, 0, 0, 0};

//*****************************************************************************
//
// IP address of target (remote) device.
//
//*****************************************************************************
uint32_t g_ui32RemoteAddress = 0;

//*****************************************************************************
//
// File name to download to target (remote) device.
//
//*****************************************************************************
char *g_pcFileName = NULL;

//*****************************************************************************
//
// A flag that is true if the application should terminate.
//
//*****************************************************************************
static bool g_bAbortMain = 0;

//*****************************************************************************
//
// This function will convert a delimited address string into an array of
// unsigned bytes.  The numbers in the address string can be delimited by
// ".", "-", or ":".  The value for each "pcToken" in the string will be stored
// into subsequent elements in the array.  The function will return the total
// number of tokens converted.
//
//*****************************************************************************
static int32_t
AddressToBytes(char *pcString, void *pValue, int32_t i32Count, int32_t i32Base)
{
    int32_t i32Converted = 0;
    char *pcToken = NULL;
    char *pcDelimit = ".-:";
    char *pcTail = NULL;

    //
    // Extract the first pcToken from the string to get the loop started.  Then,
    // For each pcToken (up to i32Count), convert it and find the next pcToken in
    // the string.  Exit the loop when i32Count has been reached, or when there
    // are no more tokens to convert.
    //
    pcToken = strtok(pcString, pcDelimit);
    while((i32Converted < i32Count) && (NULL != pcToken))
    {
        //
        // Convert the pcToken into a number.  If the conversion fails, the
        // input value of "pcTail" will match "pcToken", and that means that
        // the input string has been formatted incorrectly, so break out of
        // the process loop and simply return the number of bytes that have
        // been converted thus far.
        //
        ((uint8_t *)pValue)[i32Converted] =
            (strtoul(pcToken, &pcTail, i32Base) & 0xFF);
        if(pcTail == pcToken)
        {
            break;
        }

        //
        // Get the next pcToken and setup for the next iteration in the loop.
        //
        pcToken = strtok(NULL, pcDelimit);
        i32Converted++;
    }

    //
    // Return the number
    return(i32Converted);
}

//******************************************************************************
//
// This function will display help text in conformance with the GNU coding
// standard.
//
//******************************************************************************
static void
DisplayHelp(void)
{
    puts(g_pcProgramHelp);
}

//******************************************************************************
//
// This function will display version information in conformance with the GNU
// coding standard.
//
//******************************************************************************
static void
DisplayVersion(void)
{
    printf("%s (Version: %s)\n", g_pcProgramName, g_pcApplicationVersion);
    printf("\n");
    printf("%s\n", g_pcProgramCopyright);
    printf("\n");
}

//******************************************************************************
//
// This function will parse the command line arguments, storing any needed
// information in the appropriate variables for later reference.  The
// "getopts" command line processing library functions are used to parse
// the command line options.  The options are defined in accordance with
// the GNU coding standard.
//
//******************************************************************************
static void
ParseOptions(int32_t argc, char **argv)
{
    struct option sLongOptions[] =
    {
        //
        // GNU Standard Options that set a flag for program operation.
        //
        {"quiet",         no_argument, &g_i32OptVerbose, 0},
        {"silent",        no_argument, &g_i32OptVerbose, 0},
        {"verbose",       no_argument, &g_i32OptVerbose, 2},

        //
        // GNU Standard options that simply display information and exit.
        //
        {"help",          no_argument, 0,              0x100},
        {"version",       no_argument, 0,              0x101},

        //
        // Program specific options that set variables and/or require arguments.
        //
        {"mac",     required_argument, 0,              'm'},
        {"ip",      required_argument, 0,              'i'},

        //
        // Terminating Element of the array
        //
        {0, 0, 0, 0}
    };
    int32_t i32OptionIndex = 0;
    int32_t i32Option;
    int32_t i32ReturnCode;

    //
    // Continue parsing options till there are no more to parse.
    // Note:  The "m:i" allows the short and long options to be
    // used for the file, mac and ip parameters and will be processed
    // below by the same case statement.
    //
    while((i32Option = getopt_long(argc, argv, "m:i:", sLongOptions,
                                 &i32OptionIndex)) != -1)
    {
        //
        // Process the current option.
        //
        switch(i32Option)
        {
            //
            // Long option with flag set.
            //
            case 0:
                break;

            //
            // --help
            //
            case 0x100:
                DisplayHelp();
                exit(0);
                break;

            //
            // --version
            //
            case 0x101:
                DisplayVersion();
                exit(0);
                break;

            //
            // --mac=string, -m string
            //
           case 'm':
                i32ReturnCode = AddressToBytes(optarg, g_pui8RemoteMAC, 6, 16);
                if(i32ReturnCode != 6)
                {
                    EPRINTF(("Error Processing MAC (%d)\n", i32ReturnCode));
                    DisplayHelp();
                    exit(-(__LINE__));
                }
                break;

            //
            // --ip=string, -i string
            //
           case 'i':
                i32ReturnCode = AddressToBytes(optarg, &g_ui32RemoteAddress, 4, 10);
                if(i32ReturnCode != 4)
                {
                    EPRINTF(("Error Processing IP (%d)\n", i32ReturnCode));
                    DisplayHelp();
                    exit(-(__LINE__));
                }
                break;

            //
            // Unrecognized option.
            //
           default:
                DisplayVersion();
                DisplayHelp();
                exit(-(__LINE__));
                break;
        }
    }

    //
    // Extract filename from the last argument on the command line (if
    // provided).
    //
    if(optind == argc)
    {
        EPRINTF(("No File Name Specified\n"));
        DisplayHelp();
        exit(-(__LINE__));
    }
    else if(optind > (argc -1))
    {
        EPRINTF(("Too Many Command Line Options\n"));
        DisplayHelp();
        exit(-(__LINE__));
    }
    else
    {
        g_pcFileName = argv[optind];
    }

    //
    // Check for non-zero MAC address.
    //
    if((0 == g_pui8RemoteMAC[0]) && (0 == g_pui8RemoteMAC[1]) &&
       (0 == g_pui8RemoteMAC[2]) && (0 == g_pui8RemoteMAC[3]) &&
       (0 == g_pui8RemoteMAC[4]) && (0 == g_pui8RemoteMAC[5]))
    {
        EPRINTF(("No MAC Address Specified\n"));
        DisplayHelp();
        exit(-(__LINE__));
    }

    //
    // Check for non-zero IP address.
    //
    if(0 == g_ui32RemoteAddress)
    {
        EPRINTF(("No IP Address Specified\n"));
        DisplayHelp();
        exit(-(__LINE__));
    }
}

//*****************************************************************************
//
// A callback function to monitor the progress.
//
//*****************************************************************************
static void
StatusCallback(uint32_t ui32Percent)
{
    //
    // Print out the percentage.
    //
    if(g_i32OptVerbose == 1)
    {
        printf("%% Complete: %3d%%\r", ui32Percent);
    }
    else if(g_i32OptVerbose > 1)
    {
        printf("%% Complete: %3d%%\n", ui32Percent);
    }
}

//*****************************************************************************
//
// A callback function to monitor the progress.
//
//*****************************************************************************
static void
SignalIntHandler(int32_t i32Signal)
{
    //
    // Display a diagnostic message.
    //
    fprintf(stderr, "Abort Received (%d)... cleaning up\n", i32Signal);

    //
    // Abort the BOOTP process (if already running).
    //
    AbortBOOTPUpdate();

    //
    // Flag main to abort any processes that are running.
    //
    g_bAbortMain = 1;
}

//*****************************************************************************
//
// Main entry.  Process command line options, and start the bootp_server.
//
//*****************************************************************************
int32_t
main(int32_t argc, char **argv)
{
    HOSTENT *psHostEnt;
    WSADATA sWsaData;
    uint32_t ui32LocalAddress;

    //
    // Parse the command line options.
    //
    if(argc > 1)
    {
        ParseOptions(argc, argv);
    }
    else
    {
        DisplayVersion();
        DisplayHelp();
        return(0);
    }

    //
    // Display (if needed) verbose function entry.
    //
    if(g_i32OptVerbose > 1)
    {
        DisplayVersion();
    }

    //
    // Install an abort handler.
    //
    signal(SIGINT, SignalIntHandler);

    //
    // Startup winsock.
    //
    VPRINTF(("Starting WINSOCK\n"));
    if(WSAStartup(0x202, &sWsaData) != 0)
    {
        EPRINTF(("Winsock Failed to Start\n"));
        WSACleanup();
        return(1);
    }

    //
    // Determine what my local IP address is.
    //
    psHostEnt = gethostbyname("");
    ui32LocalAddress = ((struct in_addr *)*psHostEnt->h_addr_list)->s_addr;

    //
    // Start the BOOTP/TFTP server to perform an update.
    //
    QPRINTF(("Starting BOOTP/TFTP Server ...\n"));
    StatusCallback(0);
    StartBOOTPUpdate(g_pui8RemoteMAC, ui32LocalAddress, g_ui32RemoteAddress,
                     g_pcFileName, StatusCallback);

    //
    // Cleanup winsock.
    //
    VPRINTF(("Closing WINSOCK\n"));
    WSACleanup();

    //
    // Clean up and return.
    //
    if(g_bAbortMain)
    {
        return(2);
    }
    return(0);
}
