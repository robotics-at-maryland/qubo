//*****************************************************************************
//
// binpack.c - A simple command line tool used to wrap firmware binaries for
// download to Stellaris parts using a CRC-enabled boot loader.
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
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

//*****************************************************************************
//
// My own min macro.
//
//*****************************************************************************
#define MY_MIN(a, b)            ((a) < (b) ? (a) : (b))

//*****************************************************************************
//
// The two word markers used to indicate the start of the image information
// header in a suitably-formatted firmware image file.  This 8 word structure
// should sit at the top of the vector table.
//
//*****************************************************************************
#define INFO_MARKER0            0xFF01FF02
#define INFO_MARKER1            0xFF03FF04

//*****************************************************************************
//
// Globals controlled by various command line parameters.
//
//*****************************************************************************
bool g_bVerbose = false;
bool g_bQuiet = false;
bool g_bOverwrite = false;
bool g_bSkipHeader = true;
uint32_t g_ui32Address = 0;
uint32_t g_ui32HeaderSize = 0;
char *g_pcInput = NULL;
char *g_pcOutput = "firmware.bin";

//*****************************************************************************
//
// Helpful macros for generating output depending upon verbose and quiet flags.
//
//*****************************************************************************
#define VERBOSEPRINT(...)       if(g_bVerbose) { printf(__VA_ARGS__); }
#define QUIETPRINT(...)         if(!g_bQuiet) { printf(__VA_ARGS__); }

//*****************************************************************************
//
// Macros for accessing multi-byte fields in the file suffix.
//
//*****************************************************************************
#define WRITE_LONG(num, ptr)                                                  \
{                                                                             \
    *((uint8_t *)(ptr)) = ((num) & 0xFF);                                     \
    *(((uint8_t *)(ptr)) + 1) = (((num) >> 8) & 0xFF);                        \
    *(((uint8_t *)(ptr)) + 2) = (((num) >> 16) & 0xFF);                       \
    *(((uint8_t *)(ptr)) + 3) = (((num) >> 24) & 0xFF);                       \
}
#define WRITE_SHORT(num, ptr)                                                 \
{                                                                             \
    *((uint8_t *)(ptr)) = ((num) & 0xFF);                                     \
    *(((uint8_t *)(ptr)) + 1) = (((num) >> 8) & 0xFF);                        \
}

#define READ_SHORT(ptr)                                                       \
    (*((uint8_t *)(ptr)) | (*(((uint8_t *)(ptr)) + 1) << 8))

#define READ_LONG(ptr)                                                        \
    (*((uint8_t *)(ptr))              |                                       \
    (*(((uint8_t *)(ptr)) + 1) << 8)  |                                       \
    (*(((uint8_t *)(ptr)) + 2) << 16) |                                       \
    (*(((uint8_t *)(ptr)) + 3) << 24))

//*****************************************************************************
//
// Storage for the CRC32 calculation lookup table.
//
//*****************************************************************************
uint32_t g_pui32CRC32Table[256];

//*****************************************************************************
//
// The Tiva-specific binary image suffix used by the GUI download application
// to determine where the image should be located in flash.
//
//*****************************************************************************
uint8_t g_pui8Prefix[] =
{
    0x01, // TIVA_DFU_CMD_PROG
    0x00, // Reserved
    0x00, // LSB start address / 1024
    0x20, // MSB start address / 1024
    0x00, // LSB file payload length (excluding prefix and suffix)
    0x00, // Byte 2 file payload length (excluding prefix and suffix)
    0x00, // Byte 3 file payload length (excluding prefix and suffix)
    0x00, // MSB file payload length (excluding prefix and suffix)
};

//*****************************************************************************
//
// Initialize the CRC32 calculation table for the polynomial used.  We pick
// the commonly used ANSI X 3.66 polymonial.  This code is based on an example
// found at
//
// http://www.createwindow.com/programming/crc32/index.htm.
//
//*****************************************************************************
uint32_t
Reflect(uint32_t ui32Ref, uint8_t ui8Ch)
{
    uint32_t ui32Value, ui32Loop;

    ui32Value = 0;

    //
    // Swap bit 0 for bit 7, bit 1 for bit 6, etc.
    //
    for(ui32Loop = 1; ui32Loop < (ui8Ch + 1); ui32Loop++)
    {
        if(ui32Ref & 1)
        {
            ui32Value |= 1 << (ui8Ch - ui32Loop);
        }
        ui32Ref >>= 1;
    }

    return(ui32Value);
}

//*****************************************************************************
//
// Initialize the lookup table used in calculating the CRC32 value.
//
//*****************************************************************************
void
InitCRC32Table(void)
{
    uint32_t ui32Polynomial, ui32I, ui32J;

    //
    // This is the ANSI X 3.66 polynomial as required by the DFU
    // specification.
    //
    ui32Polynomial = 0x04c11db7;

    for(ui32I = 0; ui32I <= 0xFF; ui32I++)
    {
        g_pui32CRC32Table[ui32I] = Reflect(ui32I, 8) << 24;
        for (ui32J = 0; ui32J < 8; ui32J++)
        {
            g_pui32CRC32Table[ui32I] = ((g_pui32CRC32Table[ui32I] << 1) ^
                                        (g_pui32CRC32Table[ui32I] & (1 << 31) ?
                                         ui32Polynomial : 0));
        }
        g_pui32CRC32Table[ui32I] = Reflect(g_pui32CRC32Table[ui32I], 32);
    }
}

//*****************************************************************************
//
// Calculate the CRC for the supplied block of data.
//
//*****************************************************************************
uint32_t
CalculateCRC32(uint8_t *pui8Data, uint32_t ui32Length, uint32_t ui32CRC)
{
    uint32_t ui32Count;
    uint8_t *pui8Buffer;
    uint8_t ui8Char;

    //
    // Get a pointer to the start of the data and the number of bytes to
    // process.
    //
    pui8Buffer = pui8Data;
    ui32Count = ui32Length;

    //
    // Perform the algorithm on each byte in the supplied buffer using the
    // lookup table values calculated in InitCRC32Table().
    //
    while(ui32Count--)
    {
        ui8Char = *pui8Buffer++;
        ui32CRC = (ui32CRC >> 8) ^ g_pui32CRC32Table[(ui32CRC & 0xFF) ^ ui8Char];
    }

    // Return the result.
    return(ui32CRC);
}

//*****************************************************************************
//
// Show the startup banner.
//
//*****************************************************************************
void
PrintWelcome(void)
{
    QUIETPRINT("\nbinpack- Wrap a firmware binary file for use with a CRC-enabled boot loader.\n");
    QUIETPRINT("Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.\n\n");
}

//*****************************************************************************
//
// Show help on the application command line parameters.
//
//*****************************************************************************
void
ShowHelp(void)
{
    //
    // Only print help if we are not in quiet mode.
    //
    if(g_bQuiet)
    {
        return;
    }

    printf("This application may be used to prepare binary files which are\n");
    printf("to be flashed to a target using a CRC-enabled boot loader.\n");
    printf("Supported parameters are:\n\n");
    printf("-i <file> - The name of the input file.\n");
    printf("-o <file> - The name of the output file (default image.dfu)\n");
    printf("-a <num>  - Set the address the binary will be flashed to.  This\n"
    "                   option is required if -d is present.\n");
    printf("-d        - Adds a simple download header to the output image\n"
           "            image.  This is not needed for use with LMFlash.\n");
    printf("-x        - Overwrite existing output file without prompting.\n");
    printf("-? or -h  - Show this help.\n");
    printf("-q        - Quiet mode. Disable output to stdio.\n");
    printf("-v        - Enable verbose output\n\n");
    printf("Example:\n\n");
    printf("   binpack -i program.bin -o program_out.bin -a 0x1800\n\n");
    printf("scans program.bin for an image information header just above the\n"
           "vector table and, if found, writes the image length to the third\n"
           "word of the header and the calculated CRC32 to the fourth word.\n"
           "It then writes the resulting binary to the file program_out.bin.\n\n");
    printf("The image information header is an 8 word structure which must be\n"
           "appended to the end of the interrupt vector table by the\n"
           "application developer.  The first two words of the structure\n"
           "must be set to markers 0xFF01FF02 and 0xFF03FF04.  The third\n"
           "and fourth words are written by binpack and hold the image length\n"
           "and CRC32 respectively.  The remaining four words are currently\n"
           "reserved.\n\n");
    printf("The -d option adds a simple header to the output binary image.\n"
           "This 8 byte header contains the marker pattern 0x01, 0x00 followed\n"
           "by a uint16_t value containing the image start address divided by\n"
           "1024, and a uint32_t value containing the binary size (excluding\n"
           "the header).  The multi-byte integers are stored least significant\n"
           "byte first.\n");
}

//*****************************************************************************
//
// Parse the command line, extracting all parameters.
//
// Returns 0 on failure, 1 on success.
//
//*****************************************************************************
int
ParseCommandLine(int argc, char *argv[])
{
    int iRetcode;
    bool bRetcode;
    bool bShowHelp;

    //
    // By default, don't show the help screen.
    //
    bShowHelp = false;

    while(1)
    {
        //
        // Get the next command line parameter.
        //
        iRetcode = getopt(argc, argv, "a:i:o:dvh?qx");

        if(iRetcode == -1)
        {
            break;
        }

        switch(iRetcode)
        {
            case 'i':
            {
                g_pcInput = optarg;
                break;
            }

            case 'o':
            {
                g_pcOutput = optarg;
                break;
            }

            case 'a':
            {
                g_ui32Address = (uint32_t)strtol(optarg, NULL, 0);
                break;
            }

            case 'v':
            {
                g_bVerbose = true;
                break;
            }

            case 'd':
            {
                g_bSkipHeader = false;
                break;
            }

            case 'q':
            {
                g_bQuiet = true;
                break;
            }

            case 'x':
            {
                g_bOverwrite = true;
                break;
            }

            case '?':
            case 'h':
            {
                bShowHelp = true;
                break;
            }
        }
    }

    //
    // Show the welcome banner unless we have been told to be quiet.
    //
    PrintWelcome();

    //
    // Catch various invalid parameter cases.
    //
    if(bShowHelp || (g_pcInput == NULL) ||
      (!g_bSkipHeader && ((g_ui32Address == 0) || (g_ui32Address & 1023))))
    {
        //
        // Show the command line options.
        //
        ShowHelp();

        //
        // If we were not explicitly asked for help information, provide some
        // other help on the cause of the error.
        //
        if(!bShowHelp)
        {
            //
            // Make sure we were given an input file.
            //
            if(g_pcInput == NULL)
            {
                QUIETPRINT("ERROR: An input file must be specified using the "
                           "-i parameter.\n");
            }

            //
            // Make sure we were given a start address if we're not skipping
            // the header.
            //
            if(!g_bSkipHeader && (g_ui32Address == 0))
            {
                QUIETPRINT("ERROR: The flash address of the image must be "
                           "provided using the -a parameter.\n");
            }

            if(!g_bSkipHeader && (g_ui32Address & 1023))
            {
                QUIETPRINT("ERROR: The supplied flash address must be a "
                           "multiple of 1024.\n");
            }
        }

        //
        // If we get here, we exit immediately.
        //
        exit(1);
    }

    //
    // Tell the caller that everything is OK.
    //
    return(1);
}

//*****************************************************************************
//
// Dump the command line parameters to stdout if we are in verbose mode.
//
//*****************************************************************************
void
DumpCommandLineParameters(void)
{
    if(!g_bQuiet && g_bVerbose)
    {
        printf("Input file:        %s\n", g_pcInput);
        printf("Output file:       %s\n", g_pcOutput);
        printf("Flash Address:     0x%08x\n", g_ui32Address);
        printf("Overwrite output?: %s\n", g_bOverwrite ? "Yes" : "No");
    }
}

//*****************************************************************************
//
// Read the input file into memory, optionally leaving space before it for the
// header.
//
// On success, *pui32Length is written with the length of the buffer allocated.
// This will be the file size plus the header length if bHdrs is true.
//
// Returns a pointer to the start of the allocated buffer if successful or
// NULL if there was a problem.  The file data read starts at
// sizeof(g_pui8Prefix) bytes into the allocated block unless -s was provided
// to tell us to skip the download header structure.
//
//*****************************************************************************
uint8_t *
ReadInputFile(char *pcFilename, uint32_t *pui32Length)
{
    uint8_t *pui8FileBuffer;
    int iRead;
    int iSize;
    int iSizeAlloc;
    FILE *fhFile;

    QUIETPRINT("Reading input file %s\n", pcFilename);

    //
    // How many bytes of space do we need to leave at the beginning of the
    // file to add our download header?
    //
    g_ui32HeaderSize = (g_bSkipHeader ? 0 : sizeof(g_pui8Prefix));

    //
    // Try to open the input file.
    //
    fhFile = fopen(pcFilename, "rb");
    if(!fhFile)
    {
        //
        // File not found or cannot be opened for some reason.
        //
        QUIETPRINT("Can't open file!\n");
        return(NULL);
    }

    //
    // Determine the file length.
    //
    fseek(fhFile, 0, SEEK_END);
    iSize = ftell(fhFile);
    fseek(fhFile, 0, SEEK_SET);

    //
    // Allocate a buffer to hold the file contents and header.
    //
    iSizeAlloc = iSize + g_ui32HeaderSize;
    pui8FileBuffer = malloc(iSizeAlloc);
    if(pui8FileBuffer == NULL)
    {
        QUIETPRINT("Can't allocate %d bytes of memory!\n", iSizeAlloc);
        return(NULL);
    }

    //
    // Read the file contents into the buffer at the correct position.
    //
    VERBOSEPRINT("File size is %d (0x%x) bytes.\n", iSize, iSize);
    iRead = fread(pui8FileBuffer + g_ui32HeaderSize, 1, iSize, fhFile);

    //
    // Close the file.
    //
    fclose(fhFile);

    //
    // Did we get the whole file?
    //
    if(iSize != iRead)
    {
        //
        // Nope - free the buffer and return an error.
        //
        QUIETPRINT("Error reading file. Expected %d bytes, got %d!\n",
                   iSize, iRead);
        free(pui8FileBuffer);
        return(NULL);
    }

    //
    // Copy the header if we've been asked to add it.
    //
    if(g_ui32HeaderSize)
    {
        memcpy(pui8FileBuffer, g_pui8Prefix, sizeof(g_pui8Prefix));
    }

    //
    // Return the new buffer to the caller along with its size.
    //
    *pui32Length = (uint32_t)iSizeAlloc;
    return(pui8FileBuffer);
}

//*****************************************************************************
//
// Open the output file after checking whether it exists and getting user
// permission for an overwrite (if required) then write the supplied data to
// it.
//
// Returns 0 on success or a positive value on error.
//
//*****************************************************************************
int
WriteOutputFile(char *pcFile, uint8_t *pui8Data, uint32_t ui32Length)
{
    FILE *fh;
    int iResponse;
    uint32_t ui32Written;

    //
    // Have we been asked to overwrite an existing output file without
    // prompting?
    //
    if(!g_bOverwrite)
    {
        //
        // No - we need to check to see if the file exists before proceeding.
        //
        fh = fopen(pcFile, "rb");
        if(fh)
        {
            VERBOSEPRINT("Output file already exists.\n");

            //
            // The file already exists. Close it them prompt the user about
            // whether they want to overwrite or not.
            //
            fclose(fh);

            if(!g_bQuiet)
            {
                printf("File %s exists. Overwrite? ", pcFile);
                iResponse = getc(stdin);
                if((iResponse != 'y') && (iResponse != 'Y'))
                {
                    //
                    // The user didn't respond with 'y' or 'Y' so return an
                    // error and don't overwrite the file.
                    //
                    VERBOSEPRINT("User chose not to overwrite output.\n");
                    return(6);
                }
                printf("Overwriting existing output file.\n");
            }
            else
            {
                //
                // In quiet mode but -x has not been specified so don't
                // overwrite.
                //
                return(7);
            }
        }
    }

    //
    // If we reach here, it is fine to overwrite the file (or the file doesn't
    // already exist) so go ahead and open it.
    //
    fh = fopen(pcFile, "wb");
    if(!fh)
    {
        QUIETPRINT("Error opening output file for writing\n");
        return(8);
    }

    //
    // Write the supplied data to the file.
    //
    VERBOSEPRINT("Writing %d (0x%x) bytes to output file.\n", ui32Length,
                 ui32Length);
    ui32Written = fwrite(pui8Data, 1, ui32Length, fh);

    //
    // Close the file.
    //
    fclose(fh);

    //
    // Did we write all the data?
    //
    if(ui32Written != ui32Length)
    {
        QUIETPRINT("Error writing data to output file!  Wrote %d, "
                   "requested %d\n", ui32Written, ui32Length);
        return(9);
    }
    else
    {
        QUIETPRINT("Output file written successfully.\n");
    }

    return(0);
}

//*****************************************************************************
//
// Main entry function for the application.
//
//*****************************************************************************
uint32_t
FindImageInfoHeader(uint8_t *pui8File, uint32_t ui32Len)
{
    uint32_t ui32Check, ui32End;

    //
    // If the length is less than 32 bytes, there can't be a header so fail
    // immediately.
    //
    if(ui32Len < 32)
    {
        return(0);
    }

    //
    // We check the first 257 words of the image for the header or, if the
    // image is shorter than this, the entire image up to the the last position
    // the header may start and still leave 32 bytes for the content.
    //
    ui32End = MY_MIN(ui32Len - 32, (257 * 4));

    for(ui32Check = 0; ui32Check <= ui32End; ui32Check += 4)
    {
        if((READ_LONG(&pui8File[ui32Check]) == INFO_MARKER0) &&
           (READ_LONG(&pui8File[ui32Check + 4]) == INFO_MARKER1))
        {
            //
            // Return the position of the first header payload word.
            //
            return(ui32Check + 8);
        }
    }

    //
    // If we get here, we didn't find the header so return 0 to indicate an
    // error.
    //
    return(0);
}

//*****************************************************************************
//
// Main entry function for the application.
//
//*****************************************************************************
int
main(int argc, char *argv[])
{
    int iRetcode;
    uint8_t *pui8Input;
    uint8_t *pui8Prefix;
    uint32_t ui32FileLen;
    uint32_t ui32CRC, ui32CRCOffset, ui32LenOffset;
    bool bPrefixValid;

    //
    // Initialize the CRC32 lookup table.
    //
    InitCRC32Table();

    //
    // Parse the command line arguments
    //
    iRetcode = ParseCommandLine(argc, argv);
    if(!iRetcode)
    {
        return(1);
    }

    //
    // Echo the command line settings to stdout in verbose mode.
    //
    DumpCommandLineParameters();

    //
    // Read the input file into memory.
    //
    pui8Input = ReadInputFile(g_pcInput, &ui32FileLen);
    if(!pui8Input)
    {
        VERBOSEPRINT("Error reading input file.\n");
        exit(1);
    }

    //
    // Find the image information header at the top of the image
    // vector table.
    //
    ui32LenOffset = FindImageInfoHeader(pui8Input + g_ui32HeaderSize,
                                      ui32FileLen - g_ui32HeaderSize);

    if(!ui32LenOffset)
    {
        QUIETPRINT("Error: Invalid input image format.\n")
        QUIETPRINT("The input file contains no image info header at the top "
                   "of the vector table!\n")
        QUIETPRINT("Please ensure this is present before running this tool.\n");
        free(pui8Input);
        exit(2);
    }

    //
    // Fill in the output file header address and length fields if the output
    // file has a download header appended.
    //
    if(g_ui32HeaderSize)
    {
        WRITE_SHORT(g_ui32Address / 1024, pui8Input + 2);
        WRITE_LONG(ui32FileLen - g_ui32HeaderSize, pui8Input + 4);
    }

    //
    // Write the file length at the relevant offset within the binary.  This
    // is needed by the boot loader to allow it to check the CRC at boot time.
    //
    WRITE_LONG(ui32FileLen - g_ui32HeaderSize, pui8Input + g_ui32HeaderSize +
               ui32LenOffset);

    //
    // Calculate the new file CRC.  This is calculated over the entire
    // binary including the inserted length field but excluding the 4 bytes
    // that will contain the CRC itself.
    //
    ui32CRCOffset = ui32LenOffset + 4;
    ui32CRC = CalculateCRC32(pui8Input + g_ui32HeaderSize, ui32CRCOffset,
            0xffffffff);
    VERBOSEPRINT("First CRC portion, %d bytes from offset %d. CRC 0x%08x.\n",
                 ui32CRCOffset, g_ui32HeaderSize, ui32CRC);
    ui32CRC = CalculateCRC32(pui8Input + g_ui32HeaderSize + ui32CRCOffset + 4,
            ui32FileLen - (ui32CRCOffset + 4 + g_ui32HeaderSize), ui32CRC);
    ui32CRC ^= 0xffffffff;
    VERBOSEPRINT("Final CRC portion, %d bytes from offset %d. CRC 0x%08x.\n",
                 ui32FileLen - (ui32CRCOffset + 4 + g_ui32HeaderSize),
                 ui32CRCOffset + 4 + g_ui32HeaderSize, ui32CRC);
    WRITE_LONG(ui32CRC, pui8Input + g_ui32HeaderSize + ui32CRCOffset);

    //
    // Now write the wrapped file to the output.
    //
    iRetcode = WriteOutputFile(g_pcOutput, pui8Input, ui32FileLen);

    //
    // Free our file buffer.
    //
    free(pui8Input);

    //
    // Exit the program and tell the OS that all is well.
    //
    exit(iRetcode);
}
