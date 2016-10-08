//*****************************************************************************
//
// pnmtoc.c - Program to convert a NetPBM image to a C array.
//
// Copyright (c) 2008-2016 Texas Instruments Incorporated.  All rights reserved.
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

#include <inttypes.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//*****************************************************************************
//
// The set of colors that have been identified in the image.
//
//*****************************************************************************
uint32_t g_pui32Colors[256];

//*****************************************************************************
//
// The number of colors that have been identified in the image.
//
//*****************************************************************************
uint32_t g_ui32NumColors;

//*****************************************************************************
//
// Compares two colors based on their grayscale intensity.  This is used by
// qsort() to sort the color palette.
//
//*****************************************************************************
int32_t
ColorComp(const void *pvA, const void *pvB)
{
    int32_t ui32A, ui32B;

    //
    // Extract the grayscale value for the two colors.
    //
    ui32A = ((30 * ((*(uint32_t *)pvA >> 16) & 255)) +
             (59 * ((*(uint32_t *)pvA >> 8) & 255)) +
             (11 * (*(uint32_t *)pvA & 255)));
    ui32B = ((30 * ((*(uint32_t *)pvB >> 16) & 255)) +
             (59 * ((*(uint32_t *)pvB >> 8) & 255)) +
             (11 * (*(uint32_t *)pvB & 255)));

    //
    // Return a value that indicates their relative ordering.
    //
    return(ui32A - ui32B);
}

//*****************************************************************************
//
// Extracts the unique colors from the input image, resulting in the palette
// for the image.
//
//*****************************************************************************
void
GetNumColors(uint8_t *pui8Data, uint32_t ui32Width, uint32_t ui32Height,
             bool bMono)
{
    uint32_t ui32Count, ui32Color, ui32Idx;

    //
    // Loop through the pixels of the image.
    //
    g_ui32NumColors = 0;
    for(ui32Count = 0; ui32Count < (ui32Width * ui32Height); ui32Count++)
    {
        //
        // Extract the color of this pixel.
        //
        if(bMono)
        {
            //
            // For a mono pixel, just set the R, G and B components to the
            // sample value from the file.
            //
            ui32Color = ((pui8Data[ui32Count] << 16) |
                         (pui8Data[ui32Count] << 8) |
                         pui8Data[ui32Count]);
        }
        else
        {
            ui32Color = ((pui8Data[ui32Count * 3] << 16) |
                         (pui8Data[(ui32Count * 3) + 1] << 8) |
                         pui8Data[(ui32Count * 3) + 2]);
        }

        //
        // Loop through the current palette to see if this color is already in
        // the palette.
        //
        for(ui32Idx = 0; ui32Idx < g_ui32NumColors; ui32Idx++)
        {
            if(g_pui32Colors[ui32Idx] == ui32Color)
            {
                break;
            }
        }

        //
        // See if this color is already in the palette.
        //
        if(ui32Idx == g_ui32NumColors)
        {
            //
            // If there are already 256 colors in the palette, then there is no
            // room for this color.  Simply return and indicate that the
            // palette is too big.
            //
            if(g_ui32NumColors == 256)
            {
                g_ui32NumColors = 256 * 256 * 256;
                return;
            }

            //
            // Add this color to the palette.
            //
            g_pui32Colors[g_ui32NumColors++] = ui32Color;
        }
    }

    //
    // Sort the palette entries based on their grayscale intensity.
    //
    qsort(g_pui32Colors, g_ui32NumColors, sizeof(uint32_t), ColorComp);
}

//*****************************************************************************
//
// Encodes the image using 1 bit per pixel (providing two colors).
//
//*****************************************************************************
uint32_t
Encode1BPP(uint8_t *pui8Data, uint32_t ui32Width, uint32_t ui32Height,
           bool bMono)
{
    uint32_t ui32X, ui32Color, ui32Idx, ui32Byte, ui32Bit, ui32Length;
    uint8_t *pui8Output;

    //
    // Set the output pointer to the beginning of the input data.
    //
    pui8Output = pui8Data;

    //
    // Loop through the rows of the image.
    //
    ui32Length = 0;
    while(ui32Height--)
    {
        //
        // Loop through the output bytes of this row.
        //
        for(ui32X = 0; ui32X < ui32Width; ui32X += 8)
        {
            //
            // Loop through the columns in this byte.
            //
            for(ui32Bit = 8, ui32Byte = 0; ui32Bit > 0; ui32Bit--)
            {
                //
                // See if this column exists in the image.
                //
                if((ui32X + (8 - ui32Bit)) < ui32Width)
                {
                    //
                    // Extract the color of this pixel.
                    //
                    if(bMono)
                    {
                        //
                        // For a mono pixel, just set the R, G and B components
                        // to the sample value from the file.
                        //
                        ui32Color = ((*pui8Data << 16) | (*pui8Data << 8) |
                                     *pui8Data);
                        pui8Data++;
                    }
                    else
                    {
                        //
                        // Extract the 3 components for this color pixel
                        //
                        ui32Color = *pui8Data++ << 16;
                        ui32Color |= *pui8Data++ << 8;
                        ui32Color |= *pui8Data++;
                    }

                    //
                    // Select the entry of the palette that matches this pixel.
                    //
                    for(ui32Idx = 0; ui32Idx < g_ui32NumColors; ui32Idx++)
                    {
                        if(g_pui32Colors[ui32Idx] == ui32Color)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    //
                    // This column does not exist in the image, so provide a
                    // zero padding bit in its place.
                    //
                    ui32Idx = 0;
                }

                //
                // Insert this bit into the byte.
                //
                ui32Byte |= ui32Idx << (ui32Bit - 1);
            }

            //
            // Store this byte into the output image.
            //
            *pui8Output++ = ui32Byte;
            ui32Length++;
        }
    }

    //
    // Return the number of bytes in the encoded output.
    //
    return(ui32Length);
}

//*****************************************************************************
//
// Encodes the image using 4 bits per pixel (providing up to 16 colors).
//
//*****************************************************************************
uint32_t
Encode4BPP(uint8_t *pui8Data, uint32_t ui32Width, uint32_t ui32Height,
           bool bMono)
{
    uint32_t ui32X, ui32Color, ulIdx1, ulIdx2, ui32Count, ui32Length;
    uint8_t *pui8Output;

    //
    // Set the output pointer to the beginning of the input data.
    //
    pui8Output = pui8Data;

    //
    // Loop through the rows of the image.
    //
    ui32Length = 0;
    while(ui32Height--)
    {
        //
        // Loop through the output bytes of this row.
        //
        for(ui32X = 0; ui32X < ui32Width; ui32X += 2)
        {
            //
            // Extract the color of this pixel.
            //
            if(bMono)
            {
                //
                // For a mono pixel, just set the R, G and B components
                // to the sample value from the file.
                //
                ui32Color = ((*pui8Data << 16) | (*pui8Data << 8) |
                             *pui8Data);
                pui8Data++;
            }
            else
            {
                //
                // Extract the 3 components for this color pixel
                //
                ui32Color = *pui8Data++ << 16;
                ui32Color |= *pui8Data++ << 8;
                ui32Color |= *pui8Data++;
            }

            //
            // Select the entry of the palette that matches this pixel.
            //
            for(ulIdx1 = 0; ulIdx1 < g_ui32NumColors; ulIdx1++)
            {
                if(g_pui32Colors[ulIdx1] == ui32Color)
                {
                    break;
                }
            }

            //
            // See if the second pixel exists in the image.
            //
            if((ui32X + 1) == ui32Width)
            {
                //
                // The second pixel does not exist in the image, so provide a
                // zero padding nibble in its place.
                //
                ulIdx2 = 0;
            }
            else
            {
                //
                // Extract the color of the second pixel.
                //
                if(bMono)
                {
                    //
                    // For a mono pixel, just set the R, G and B components
                    // to the sample value from the file.
                    //
                    ui32Color = ((*pui8Data << 16) | (*pui8Data << 8) |
                                 *pui8Data);
                    pui8Data++;
                }
                else
                {
                    //
                    // Extract the 3 components for this color pixel
                    //
                    ui32Color = *pui8Data++ << 16;
                    ui32Color |= *pui8Data++ << 8;
                    ui32Color |= *pui8Data++;
                }

                //
                // Select the entry of the palette that matches this pixel.
                //
                for(ulIdx2 = 0; ulIdx2 < g_ui32NumColors; ulIdx2++)
                {
                    if(g_pui32Colors[ulIdx2] == ui32Color)
                    {
                        break;
                    }
                }
            }

            //
            // Combine the two nibbles and store the byte into the output
            // image.
            //
            *pui8Output++ = (ulIdx1 << 4) | ulIdx2;
            ui32Length++;
        }
    }

    //
    // Return the number of bytes in the encoded output.
    //
    return(ui32Length);
}

//*****************************************************************************
//
// Encodes the image using 8 bits per pixel (providing up to 256 colors).
//
//*****************************************************************************
uint32_t
Encode8BPP(uint8_t *pui8Data, uint32_t ui32Width, uint32_t ui32Height,
           bool bMono)
{
    uint32_t ui32X, ui32Color, ui32Idx, ui32Length;
    uint8_t *pui8Output;

    //
    // Set the output pointer to the beginning of the input data.
    //
    pui8Output = pui8Data;

    //
    // Loop through the rows of the image.
    //
    ui32Length = 0;
    while(ui32Height--)
    {
        //
        // Loop through the columns of the image.
        //
        for(ui32X = 0; ui32X < ui32Width; ui32X++)
        {
            //
            // Extract the color of this pixel.
            //
            if(bMono)
            {
                //
                // For a mono pixel, just set the R, G and B components
                // to the sample value from the file.
                //
                ui32Color = ((*pui8Data << 16) | (*pui8Data << 8) |
                             *pui8Data);
                pui8Data++;
            }
            else
            {
                //
                // Extract the 3 components for this color pixel
                //
                ui32Color = *pui8Data++ << 16;
                ui32Color |= *pui8Data++ << 8;
                ui32Color |= *pui8Data++;
            }

            //
            // Select the entry of the palette that matches this pixel.
            //
            for(ui32Idx = 0; ui32Idx < g_ui32NumColors; ui32Idx++)
            {
                if(g_pui32Colors[ui32Idx] == ui32Color)
                {
                    break;
                }
            }

            //
            // Store the byte into the output image.
            //
            *pui8Output++ = ui32Idx;
            ui32Length++;
        }
    }

    //
    // Return the number of bytes in the encoded output.
    //
    return(ui32Length);
}

//*****************************************************************************
//
// Compresses the image data using the Lempel-Ziv-Storer-Szymanski compression
// algorithm.
//
//*****************************************************************************
uint32_t
CompressData(uint8_t *pui8Data, uint32_t ui32Length)
{
    uint8_t pui8Dictionary[32], *pui8Output, *pui8Ptr, ui8Bits, pui8Encode[9];
    uint32_t ui32EncodedLength, ui32Idx, ui32Size, ui32Match, ui32MatchLen;
    uint32_t ui32Count;

    //
    // Allocate a buffer to hold the compressed output.  In certain cases, the
    // "compressed" output may be larger than the input data.  In all cases,
    // the first several bytes of the compressed output will be larger than the
    // input data, making an in-place compression impossible.
    //
    pui8Output = pui8Ptr = malloc(((ui32Length * 9) + 7) / 8);

    //
    // Clear the dictionary.
    //
    memset(pui8Dictionary, 0, sizeof(pui8Dictionary));

    //
    // Reset the state of the encoded sequence.
    //
    ui8Bits = 0;
    pui8Encode[0] = 0;
    ui32EncodedLength = 0;

    //
    // Loop through the input data.
    //
    for(ui32Count = 0; ui32Count < ui32Length; ui32Count++)
    {
        //
        // Loop through the current dictionary.
        //
        for(ui32Idx = 0, ui32MatchLen = 0; ui32Idx < sizeof(pui8Dictionary);
            ui32Idx++)
        {
            //
            // See if this input byte matches this byte of the dictionary.
            //
            if(pui8Dictionary[ui32Idx] == pui8Data[ui32Count])
            {
                //
                // Loop through the next bytes of the input, comparing them to
                // the next bytes of the dictionary.  This determines the
                // length of this match of this portion of the input data to
                // this portion of the dictonary.
                //
                for(ui32Size = 1;
                    (ui32Idx + ui32Size) < sizeof(pui8Dictionary); ui32Size++)
                {
                    if(pui8Dictionary[ui32Idx + ui32Size] !=
                       pui8Data[ui32Count + ui32Size])
                    {
                        break;
                    }
                }

                //
                // If the match is at least three bytes (since one or two bytes
                // can be encoded just as well or better using a literal
                // encoding instead of a dictionary reference) and the match is
                // longer than any previously found match, then remember the
                // position and length of this match.
                //
                if((ui32Size > 2) && (ui32Size > ui32MatchLen))
                {
                    ui32Match = ui32Idx;
                    ui32MatchLen = ui32Size;
                }
            }
        }

        //
        // See if a match was found.
        //
        if(ui32MatchLen != 0)
        {
            //
            // The maximum length match that can be encoded is 9 bytes, so
            // limit the match length if required.
            //
            if(ui32MatchLen > 9)
            {
                ui32MatchLen = 9;
            }

            //
            // Indicate that this byte of the encoded data is a dictionary
            // reference.
            //
            pui8Encode[0] |= (1 << (7 - ui8Bits));

            //
            // Save the dictionary reference in the encoded data stream.
            //
            pui8Encode[ui8Bits + 1] = (ui32Match << 3) | (ui32MatchLen - 2);

            //
            // Shift the dictionary by the number of bytes in the match and
            // copy that many bytes from the input data stream into the
            // dictionary.
            //
            memcpy(pui8Dictionary, pui8Dictionary + ui32MatchLen,
                   sizeof(pui8Dictionary) - ui32MatchLen);
            memcpy(pui8Dictionary + sizeof(pui8Dictionary) - ui32MatchLen,
                   pui8Data + ui32Count, ui32MatchLen);

            //
            // Increment the count of input bytes consumed by the size of the
            // dictionary match.
            //
            ui32Count += ui32MatchLen - 1;
        }
        else
        {
            //
            // Save the literal byte in the encoded data stream.
            //
            pui8Encode[ui8Bits + 1] = pui8Data[ui32Count];

            //
            // Shift the dictionary by the single literal byte and copy that
            // byte from the input stream into the dictionary.
            //
            memcpy(pui8Dictionary, pui8Dictionary + 1,
                   sizeof(pui8Dictionary) - 1);
            pui8Dictionary[sizeof(pui8Dictionary) - 1] = pui8Data[ui32Count];
        }

        //
        // Increment the count of flag bits that have been used.
        //
        ui8Bits++;

        //
        // See if all eight flag bits have been used.
        //
        if(ui8Bits == 8)
        {
            //
            // Copy this 9 byte encoded sequence to the output.
            //
            memcpy(pui8Ptr, pui8Encode, 9);
            pui8Ptr += 9;
            ui32EncodedLength += 9;

            //
            // Reset the encoded sequence state.
            //
            ui8Bits = 0;
            pui8Encode[0] = 0;
        }
    }

    //
    // See if there is any residual data left in the encoded sequence buffer.
    //
    if(ui8Bits != 0)
    {
        //
        // Copy the residual data from the encoded sequence buffer to the
        // output.
        //
        memcpy(pui8Ptr, pui8Encode, ui8Bits + 1);
        ui32EncodedLength += ui8Bits + 1;
    }

    //
    // If the encoded length of the data is larger than the unencoded length of
    // the data, then discard the encoded data.
    //
    if(ui32EncodedLength > ui32Length)
    {
        free(pui8Output);
        return(ui32Length);
    }

    //
    // Coyp the encoded data to the input data buffer.
    //
    memcpy(pui8Data, pui8Output, ui32EncodedLength);

    //
    // Free the temporary encoded data buffer.
    //
    free(pui8Output);

    //
    // Return the length of the encoded data, setting the flag in the size that
    // indicates that the data is encoded.
    //
    return(ui32EncodedLength | 0x80000000);
}

//*****************************************************************************
//
// Prints a C array definition corresponding to the processed image.
//
//*****************************************************************************
void
OutputData(uint8_t *pui8Data, uint32_t ui32Width, uint32_t ui32Height,
           uint32_t ui32Length)
{
    uint32_t ui32Idx, ui32Count;

    //
    // Print the image header.
    //
    printf("const uint8_t g_pui8Image[] =\n");
    printf("{\n");

    //
    // Print the image format based on the number of colors and the use of
    // compression.
    //
    if(g_ui32NumColors <= 2)
    {
        if(ui32Length & 0x80000000)
        {
            printf("    IMAGE_FMT_1BPP_COMP,\n");
        }
        else
        {
            printf("    IMAGE_FMT_1BPP_UNCOMP,\n");
        }
    }
    else if(g_ui32NumColors <= 16)
    {
        if(ui32Length & 0x80000000)
        {
            printf("    IMAGE_FMT_4BPP_COMP,\n");
        }
        else
        {
            printf("    IMAGE_FMT_4BPP_UNCOMP,\n");
        }
    }
    else
    {
        if(ui32Length & 0x80000000)
        {
            printf("    IMAGE_FMT_8BPP_COMP,\n");
        }
        else
        {
            printf("    IMAGE_FMT_8BPP_UNCOMP,\n");
        }
    }

    //
    // Strip the compression flag from the image length.
    //
    ui32Length &= 0x7fffffff;

    //
    // Print the width and height of the image.
    //
    printf("    %" PRIu32 ", %" PRIu32 ",\n", ui32Width & 255,
           ui32Width / 256);
    printf("    %" PRIu32 ", %" PRIu32 ",\n", ui32Height & 255,
           ui32Height / 256);
    printf("\n");

    //
    // For 4 and 8 bit per pixel formats, print out the color palette.
    //
    if(g_ui32NumColors > 2)
    {
        printf("    %" PRIu32 ",\n", g_ui32NumColors - 1);
        for(ui32Idx = 0; ui32Idx < g_ui32NumColors; ui32Idx++)
        {
            printf("    0x%02" PRIx32 ", 0x%02" PRIx32 ", 0x%02" PRIx32
                   ",\n", g_pui32Colors[ui32Idx] & 255,
                   (g_pui32Colors[ui32Idx] >> 8) & 255,
                   (g_pui32Colors[ui32Idx] >> 16) & 255);
        }
        printf("\n");
    }

    //
    // Loop through the image data bytes.
    //
    for(ui32Idx = 0, ui32Count = 0; ui32Idx < ui32Length; ui32Idx++)
    {
        //
        // If this is the first byte of an output line, then provide the
        // required indentation.
        //
        if(ui32Count++ == 0)
        {
            printf("   ");
        }

        //
        // Print the value of this data byte.
        //
        printf(" 0x%02x,", pui8Data[ui32Idx]);

        //
        // If this is the last byte on an output line, then output a newline.
        //
        if(ui32Count == 12)
        {
            printf("\n");
            ui32Count = 0;
        }
    }

    //
    // If a partial line of bytes has been output, then output a newline.
    //
    if(ui32Count != 0)
    {
        printf("\n");
    }

    //
    // Close the array definition.
    //
    printf("};\n");
}

//*****************************************************************************
//
// Prints the usage message for this application.
//
//*****************************************************************************
void
Usage(char *pucProgram)
{
    fprintf(stderr, "Usage: %s [OPTION] [FILE]\n", basename(pucProgram));
    fprintf(stderr, "Converts a Netpbm file to a C array for use by the "
            "TivaWare Graphics Library.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  -c  Compresses the image using Lempel-Ziv-Storer-"
            "Szymanski\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "The image format is chosen based on the number of colors "
            "in the image; for\n");
    fprintf(stderr, "example, if there are 12 colors in the image, the 4BPP "
            "image format is used.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Report bugs to <support_lmi@ti.com>.\n");
}

//*****************************************************************************
//
// The main application that converts a NetPBM image into a C array for use by
// the TivaWare Graphics Library.
//
//*****************************************************************************
int
main(int argc, char *argv[])
{
    uint32_t ui32Length, ui32Width, ui32Height, ui32Max, ui32Count;
    uint8_t *pui8Data;
    int32_t i32Opt, i32Compress;
    bool bMono, bBitmap;
    FILE *pFile;

    //
    // Compression of the image is off by default.
    //
    i32Compress = 0;

    //
    // Loop through the switches found on the command line.
    //
    while((i32Opt = getopt(argc, argv, "ch")) != -1)
    {
        //
        // Determine which switch was identified.
        //
        switch(i32Opt)
        {
            //
            // The "-c" switch was found.
            //
            case 'c':
            {
                //
                // Enable compression of the image.
                //
                i32Compress = 1;

                //
                // This switch has been handled.
                //
                break;
            }

            //
            // The "-h" switch, or an unknown switch, was found.
            //
            case 'h':
            default:
            {
                //
                // Display the usage information.
                //
                Usage(argv[0]);

                //
                // Return an error.
                //
                return(1);
            }
        }
    }

    //
    // There must be one additional argument on the command line, which
    // provides the name of the image file.
    //
    if((optind + 1) != argc)
    {
        //
        // Display the usage information.
        //
        Usage(argv[0]);

        //
        // Return an error.
        //
        return(1);
    }

    //
    // Open the input image file.
    //
    pFile = fopen(argv[optind], "rb");
    if(!pFile)
    {
        fprintf(stderr, "%s: Unable to open input file '%s'\n",
                basename(argv[0]), argv[optind]);
        return(1);
    }

    //
    // Get the length of the input image file.
    //
    fseek(pFile, 0, SEEK_END);
    ui32Length = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);

    //
    // Allocate a memory buffer to store the input image file.
    //
    pui8Data = malloc(ui32Length);
    if(!pui8Data)
    {
        fprintf(stderr, "%s: Unable to allocate buffer for file.\n",
                basename(argv[0]));
        return(1);
    }

    //
    // Read the input image file into the memory buffer.
    //
    if(fread(pui8Data, 1, ui32Length, pFile) != ui32Length)
    {
        fprintf(stderr, "%s: Unable to read file data.\n", basename(argv[0]));
        return(1);
    }

    //
    // Close the input image file.
    //
    fclose(pFile);

    //
    // Parse the file header to extract the width and height of the image, and
    // to verify that the image file is a format that is recognized.
    //
    if((pui8Data[0] != 'P') || ((pui8Data[1] != '4') && (pui8Data[1] != '6') &&
                                (pui8Data[1] != '5')))
    {
        fprintf(stderr, "%s: '%s' is not a supported PNM file.\n",
                basename(argv[0]), argv[1]);
        return(1);
    }

    //
    // Are we dealing with a color or grayscale image?
    //
    bMono = (pui8Data[1] == '5') ? 1 : 0;

    //
    // Are we dealing with a (1bpp) bitmap?
    //
    bBitmap = (pui8Data[1] == '4') ? 1 : 0;

    //
    // Loop through the values to be read from the header.
    //
    for(ui32Count = 0, ui32Max = 2; ui32Count < 3; )
    {
        //
        // Loop until a non-whitespace character is found.
        //
        while((pui8Data[ui32Max] == ' ') || (pui8Data[ui32Max] == '\t') ||
              (pui8Data[ui32Max] == '\r') || (pui8Data[ui32Max] == '\n'))
        {
            ui32Max++;
        }

        //
        // If this is a '#', then it starts a comment line.  Ignore comment
        // lines.
        //
        if(pui8Data[ui32Max] == '#')
        {
            //
            // Loop until the end of the line is found.
            //
            while((pui8Data[ui32Max] != '\r') && (pui8Data[ui32Max] != '\n'))
            {
                ui32Max++;
            }

            //
            // Restart the loop.
            //
            continue;
        }

        //
        // Read this value from the file.
        //
        if(ui32Count == 0)
        {
            if(sscanf(pui8Data + ui32Max, "%" PRIu32, &ui32Width) != 1)
            {
                fprintf(stderr, "%s: '%s' has an invalid width.\n",
                        basename(argv[0]), argv[1]);
                return(1);
            }
        }
        else if(ui32Count == 1)
        {
            if(sscanf(pui8Data + ui32Max, "%" PRIu32, &ui32Height) != 1)
            {
                fprintf(stderr, "%s: '%s' has an invalid height.\n",
                        basename(argv[0]), argv[1]);
                return(1);
            }

            //
            // We've finished reading the header if this is a bitmap so force
            // the loop to exit.
            //
            if(bBitmap)
            {
                ui32Count = 2;
            }
        }
        else
        {
            //
            // Read the maximum number of colors.  We ignore this value but
            // need to skip past it.  Note that, if this is a bitmap, we will
            // never get here.
            //
            if(sscanf(pui8Data + ui32Max, "%" PRIu32, &ui32Count) != 1)
            {
                fprintf(stderr, "%s: '%s' has an invalid maximum value.\n",
                        basename(argv[0]), argv[1]);
                return(1);
            }
            ui32Count = 2;
        }
        ui32Count++;

        //
        // Skip past this value.
        //
        while((pui8Data[ui32Max] != ' ') && (pui8Data[ui32Max] != '\t') &&
              (pui8Data[ui32Max] != '\r') && (pui8Data[ui32Max] != '\n'))
        {
            ui32Max++;
        }
    }

    //
    // Find the end of this line.
    //
    while((pui8Data[ui32Max] != '\r') && (pui8Data[ui32Max] != '\n'))
    {
        ui32Max++;
    }

    //
    // Skip this end of line marker.
    //
    ui32Max++;
    if((pui8Data[ui32Max] == '\r') || (pui8Data[ui32Max] == '\n'))
    {
        ui32Max++;
    }

    //
    // Is this a bitmap?
    //
    if(!bBitmap)
    {
        //
        // No - get the number of distinct colors in the image.
        //
        GetNumColors(pui8Data + ui32Max, ui32Width, ui32Height, bMono);

        //
        // Determine how many colors are in the image.
        //
        if(g_ui32NumColors <= 2)
        {
            //
            // There are 1 or 2 colors in the image, so encode it with 1 bit
            // per pixel.
            //
            ui32Length = Encode1BPP(pui8Data + ui32Max, ui32Width, ui32Height,
                                    bMono);
        }
        else if(g_ui32NumColors <= 16)
        {
            //
            // There are 3 through 16 colors in the image, so encode it with
            // 4 bits per pixel.
            //
            ui32Length = Encode4BPP(pui8Data + ui32Max, ui32Width, ui32Height,
                                    bMono);
        }
        else if(g_ui32NumColors <= 256)
        {
            //
            // There are 17 through 256 colors in the image, so encode it with
            // 8 bits per pixel.
            //
            ui32Length = Encode8BPP(pui8Data + ui32Max, ui32Width, ui32Height,
                                    bMono);
        }
        else
        {
            //
            // There are more than 256 colors in the image, which is not
            // supported.
            //
            fprintf(stderr, "%s: Image contains too many colors!\n",
                    basename(argv[0]));
            return(1);
        }
    }
    else
    {
        //
        // This is a bitmap so the palette consists of black and white only.
        //
        g_ui32NumColors = 2;

        //
        // Set up the palette needed for the data. PBM format defines 1 as a
        // black pixel and 0 as a white one but we need to invert this to
        // match the TivaWare graphics library format.
        //
        g_pui32Colors[1] = 0x00FFFFFF;
        g_pui32Colors[0] = 0x00000000;

        //
        // The data as read from the file is fine so we don't need to do any
        // reformatting now that the palette is set up.  Just set up the length
        // of the bitmap data remembering that each line is padded to a whole
        // byte.  First check that the data we read is actually the correct
        // length.
        //
        if((ui32Length - ui32Max) < (((ui32Width + 7) / 8) * ui32Height))
        {
            fprintf(stderr, "%s: Image data truncated. Size %" PRIu32 "bytes "
                    "but dimensions are %" PRIu32 " x %" PRIu32 ".\n",
                    basename(argv[0]), (ui32Length - ui32Max), ui32Width,
                    ui32Height);
            return(1);
        }

        //
        // Set the length to the expected value.
        //
        ui32Length = ((ui32Width + 7) / 8) * ui32Height;

        //
        // Invert the image data to make 1 bits foreground (white) and 0
        // bits background (black).
        //
        for(ui32Count = 0; ui32Count < ui32Length; ui32Count++)
        {
            *(pui8Data + ui32Max + ui32Count) =
                ~(*(pui8Data + ui32Max + ui32Count));
        }
    }

    //
    // Compress the image data if requested.
    //
    if(i32Compress)
    {
        ui32Length = CompressData(pui8Data + ui32Max, ui32Length);
    }

    //
    // Print the C array corresponding to the image data.
    //
    OutputData(pui8Data + ui32Max, ui32Width, ui32Height, ui32Length);

    //
    // Success.
    //
    return(0);
}
