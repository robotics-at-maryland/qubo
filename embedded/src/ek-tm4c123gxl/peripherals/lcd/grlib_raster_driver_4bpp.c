//*****************************************************************************
//
// grlib_raster_driver_4bpp.c - Generic graphics library display driver
// supporting screens attached to the LCD controller via its raster interface
// and making use of a 16-color, 4bpp display buffer.
//
// Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
// 
//   Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
//   Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the  
//   distribution.
// 
//   Neither the name of Texas Instruments Incorporated nor the names of
//   its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// This is part of revision 2.1.3.156 of the Tiva Graphics Library.
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_gpio.h"
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/rom.h"
#include "driverlib/lcd.h"
#include "driverlib/debug.h"
#include "grlib/grlib.h"
#include "drivers/grlib_raster_driver_4bpp.h"

//*****************************************************************************
//
// If using a Snowflake revision A0 device, define USE_EPI_WORKAROUND when
// compiling this file to include code that works around an EPI memory
// access erratum for this device.  Without this definition, you run the
// risk of seeing occasional pixel corruption on affected parts.
//
//*****************************************************************************
#ifdef USE_EPI_WORKAROUND
#include "driverlib/epi.h"
#define WRITE_WORD(Addr, Val)  EPIWorkaroundWordWrite((Addr), (Val))
#define WRITE_HWORD(Addr, Val) EPIWorkaroundHWordWrite((Addr), (Val))
#define WRITE_BYTE(Addr, Val)  EPIWorkaroundByteWrite((Addr), (Val))
#define READ_HWORD(Addr) EPIWorkaroundHWordRead(Addr)
#define READ_BYTE(Addr)  EPIWorkaroundByteRead(Addr)
#else
#define WRITE_WORD(Addr, Val)  HWREG(Addr) = (Val)
#define WRITE_HWORD(Addr, Val) HWREGH(Addr) = (Val)
#define WRITE_BYTE(Addr, Val)  HWREGB(Addr) = (Val)
#define READ_HWORD(Addr) HWREGH(Addr)
#define READ_BYTE(Addr)  HWREGB(Addr)
#endif

//*****************************************************************************
//
// To use this driver, the application must define several build environment
// variables as follow:
//
// RASTER_WIDTH defines the number of pixels across a single scan line of the
// display.
//
// RASTER_HEIGHT defines the number of raster lines on the display.
//
// Additionally, one of the following may be defined to control the orientation
// of the image on the display.  Note that these rotations are relative to
// the normal raster scan which is assumed to start in the top left of the
// display and scan across then down.  If none of these four labels is
// defined, the displayed image is oriented using the same origin as the
// raster.
//
// NORMAL orients the display with the same top left origin as the raster scan
// with x increasing to the right and y increasing downwards. This is the
// default if no orientation is specified.

// DISPLAY_ROTATE_90 rotates the displayed image 90 degrees clockwise with
// respect to the raster scan, placing the origin in the top right of the
// raster with x increasing down the screen and y increasing from right to
// left.
//
// DISPLAY_ROTATE_180 rotates the displayed image 180 degrees with respect to
// the raster scan, placing the origin in the bottom right of the raster with x
// increasing to the left and y increasing from bottom to top.
//
// DISPLAY_ROTATE_270 rotates the displayed image 270 degrees clockwise with
// respect to the raster scan, placing the origin in the bottom left of the
// raster with x increasing up the screen and y increasing from left to right
//
//*****************************************************************************

#ifndef RASTER_WIDTH
#error Label RASTER_WIDTH must be defined in the build environment!
#endif

#ifndef RASTER_HEIGHT
#error Label RASTER_HEIGHT must be defined in the build environment!
#endif

//*****************************************************************************
//
// If no orientation is specified, default to normal orientation where the
// application coordinate space matches the raster scan orientation.
//
//*****************************************************************************
#if !(defined NORMAL) && !(defined DISPLAY_ROTATE_90) &&                      \
    !(defined DISPLAY_ROTATE_180) && !(defined DISPLAY_ROTATE_270)
#define NORMAL
#endif

//*****************************************************************************
//
// Helpful labels relating to the layout of the palette.
//
//*****************************************************************************
#define FB_TYPE_MASK        0x7000
#define FB_TYPE_4BPP        0x2000

#define PAL_RED_MASK        0x0F00
#define PAL_GREEN_MASK      0x00F0
#define PAL_BLUE_MASK       0x000F
#define PAL_RED_SHIFT       8
#define PAL_GREEN_SHIFT     4
#define PAL_BLUE_SHIFT      0

#define GRLIB_RED_MASK      0x00FF0000
#define GRLIB_GREEN_MASK    0x0000FF00
#define GRLIB_BLUE_MASK     0x000000FF
#define GRLIB_RED_SHIFT     16
#define GRLIB_GREEN_SHIFT   8
#define GRLIB_BLUE_SHIFT    0

#define GRLIB_COLOR_TO_PAL_ENTRY(x)  (                                       \
        (((x) & 0x00F00000) >> 12) |                                         \
        (((x) & 0x0000F000) >> 8)  |                                         \
        (((x) & 0x000000F0) >> 4))

#define RED_FROM_PAL_ENTRY(x)   (((x) & PAL_RED_MASK) >> PAL_RED_SHIFT)
#define GREEN_FROM_PAL_ENTRY(x) (((x) & PAL_GREEN_MASK) >> PAL_GREEN_SHIFT)
#define BLUE_FROM_PAL_ENTRY(x)  (((x) & PAL_BLUE_MASK) >> PAL_BLUE_SHIFT)

//*****************************************************************************
//
// Driver instance data structure
//
//*****************************************************************************
typedef struct
{
    uint8_t  *pui8FrameBuffer;
    uint16_t *pui16Palette;
    uint8_t  pui8Lookup[256];
}
tRaster4bppDriverInst;

tRaster4bppDriverInst g_Raster4bppInst;

//*****************************************************************************
//
// Various definitions controlling coordinate space mapping in the four
// supported orientations.
//
//*****************************************************************************
#ifdef NORMAL
#define MAPPED_X(x, y) (x)
#define MAPPED_Y(x, y) (y)
#endif
#ifdef DISPLAY_ROTATE_90
#define MAPPED_X(x, y) ((RASTER_WIDTH - 1) - (y))
#define MAPPED_Y(x, y) (x)
#endif
#ifdef DISPLAY_ROTATE_180
#define MAPPED_X(x, y) ((RASTER_WIDTH - 1) - (x))
#define MAPPED_Y(x, y) ((RASTER_HEIGHT - 1) - (y))
#endif
#ifdef DISPLAY_ROTATE_270
#define MAPPED_X(x, y) (y)
#define MAPPED_Y(x, y) ((RASTER_HEIGHT - 1) - (x))
#endif

//*****************************************************************************
//
// Translates a 24-bit RGB color to a display driver-specific 4bpp color.
//
// \param c is the 24-bit RGB color.  The least-significant byte is the blue
// channel, the next byte is the green channel, and the third byte is the red
// channel.
//
// This macro translates a 24-bit RGB color into a value that can be written
// into the display's 4bpp frame buffer in order to reproduce that color, or
// or closest possible approximation of that color.
//
// \return Returns the display-driver specific color.
//
//*****************************************************************************
#define DPYCOLORTRANSLATE(pInst, c)   (FindClosestColorInPalette(pInst, c))

//*****************************************************************************
//
// Scans the current color palette and returns the index of the color which
// most closely matches the value passed.
//
//*****************************************************************************
static uint8_t
FindClosestColorInPalette(tRaster4bppDriverInst *pInst, uint32_t ui32RGB)
{
    uint32_t ui32Distance, ui32MinDistance, ui32MinIndex;
    uint32_t ui32Loop;
    int32_t i32Work;
    uint16_t ui16Entry;

    //
    // Set up for the search.
    //
    ui32MinDistance = 0xFFFFFFFF;
    ui32MinIndex = 0;
    ui32RGB = GRLIB_COLOR_TO_PAL_ENTRY(ui32RGB);

    //
    // Step through each palette location.
    //
    for(ui32Loop = 0; ui32Loop < 16; ui32Loop++)
    {
        //
        // Read the current palette entry.
        //
        ui16Entry = READ_HWORD(&pInst->pui16Palette[ui32Loop]);

        //
        // First check to see if there is an exact color match and, if so,
        // return immediately.
        //
        if(ui32RGB == (ui16Entry & ~FB_TYPE_MASK))
        {
            return(ui32Loop);
        }

        //
        // Calculate the squared red distance between the palette value and
        // the supplied color and set the overall distance from this.
        //
        i32Work = RED_FROM_PAL_ENTRY(ui16Entry) -
                  RED_FROM_PAL_ENTRY(ui32RGB);
        i32Work *= i32Work;
        ui32Distance = (uint32_t)i32Work;

        //
        // Calculate the squared green distance between the palette value and
        // the supplied color and update the overall distance from this.
        //
        i32Work = GREEN_FROM_PAL_ENTRY(ui16Entry) -
                  GREEN_FROM_PAL_ENTRY(ui32RGB);
        i32Work *= i32Work;
        ui32Distance += (uint32_t)i32Work;

        //
        // Calculate the squared blue distance between the palette value and
        // the supplied color and update the overall distance from this.
        //
        i32Work = BLUE_FROM_PAL_ENTRY(ui16Entry) -
                  BLUE_FROM_PAL_ENTRY(ui32RGB);
        i32Work *= i32Work;
        ui32Distance += (uint32_t)i32Work;

        //
        // Is this palette value closer to the requested color than any we
        // have yet looked at?
        //
        if(ui32Distance < ui32MinDistance)
        {
            //
            // Yes - update our closest color.
            //
            ui32MinDistance = ui32Distance;
            ui32MinIndex = ui32Loop;
        }
    }

    //
    // Return the closest color we found.
    //
    return((uint8_t)ui32MinIndex);
}

//*****************************************************************************
//
// Determines whether the passed RGB24 palette is equivalent to the palette
// currently set in the display buffer.  Returns true if the palette is
// equivalent, false otherwise.
//
// The palettes are considered equivalent if the 12-bit color in each location
// of the hardware palette matches the converted 24-bit color in the same
// location of the source palette.
//
//*****************************************************************************
static bool
PaletteIsEquivalent(tRaster4bppDriverInst *pInst,
                    const uint8_t *pui8SrcPalette, uint32_t ui32NumCols)
{
    uint32_t ui32Loop;
    uint16_t ui16Entry;

    //
    // If the source palette contains more colors than the hardware palette,
    // we have to declare them not equivalent (even though the image may only
    // contain pixels in the 0-15 range) because we have no idea what pixel
    // values are actually used at this point.
    //
    if(ui32NumCols > 16)
    {
        return(false);
    }

    //
    // Loop through as many colors as we need to look at.
    //
    for(ui32Loop = 0; ui32Loop < ui32NumCols; ui32Loop++)
    {
        //
        // Does the RGB24 color in the source palette match the color in the
        // same location in the frame buffer palette?
        //
        ui16Entry = READ_HWORD(&pInst->pui16Palette[ui32Loop]);
        if(GRLIB_COLOR_TO_PAL_ENTRY(*(uint32_t *)pui8SrcPalette) !=
           (ui16Entry & ~FB_TYPE_MASK))
        {
            return(false);
        }

        //
        // Move on to the next palette location.
        //
        pui8SrcPalette += 3;
    }

    //
    // If we get here, the palettes are equivalent.
    //
    return(true);
}

//*****************************************************************************
//
// Generates a lookup table that maps pixels from one palette to the current
// hardware frame buffer palette.  This is used in the PixelDrawMultiple call
// where a lookup table is used rather than scanning the whole palette to
// find a matching color for each and every pixel.
//
//*****************************************************************************
static void
GenerateImagePaletteLookup(tRaster4bppDriverInst *pInst,
                           const uint8_t *pui8SrcPalette, uint32_t ui32NumCols)
{
    uint32_t ui32Loop;

    //
    // If the palettes are equivalent, the lookup table is a trivial mapping.
    //
    if(PaletteIsEquivalent(pInst, pui8SrcPalette, ui32NumCols))
    {
        //
        // The palettes are the same so our lookup table is a 1:1 mapping.
        //
        for(ui32Loop = 0; ui32Loop < ui32NumCols; ui32Loop++)
        {
            pInst->pui8Lookup[ui32Loop] = (uint8_t)ui32Loop;
        }
    }
    else
    {
        //
        // The palette is not equivalent so we need to build a lookup table.
        // Loop through each possible source palette entry.
        //
        for(ui32Loop = 0; ui32Loop < ui32NumCols; ui32Loop++)
        {
            //
            // Find the hardware palette entry closest to this source palette
            // color.
            //
            pInst->pui8Lookup[ui32Loop] = FindClosestColorInPalette(pInst,
                                                   *(uint32_t *)pui8SrcPalette);

            //
            // Move to the next entry in the source palette.
            //
            pui8SrcPalette += 3;
        }
    }
}

//*****************************************************************************
//
// Configures the 4bpp raster display driver.
//
// \param pui32FrameBuffer points to the first byte of the frame buffer
// memory.  This corresponds to the first byte of the palette header for
// the frame buffer.
//
// This function must be called to initialize the 4bpp raster display driver
// for use with the Stellaris Graphics Library and the supplied frame buffer.
// The caller is responsible for allocating frame buffer memory and
// configuring the LCD controller appropriately prior to calling this function.
//
// The frame buffer provided must conform to the requirements imposed by the
// LCD controller raster engine. The buffer starts with a header containing
// the color palette and pixel format information. This header is followed
// by the \b RASTER_WIDTH * \b RASTER_HEIGHT pixel array.  The palette
// contains 16 half words for a 4bpp frame buffer format.
//
// Although the display raster scans in a fixed direction with respect to a
// given origin, the raster display driver can be configured to rotate the
// displayed image to allow displays to be viewed in any portrait or landscape
// format.  The orientation is selected at build time by defining one of
// \b NORMAL which orients the displayed image with the
// same origin as the raster scan, \b DISPLAY_ROTATE_90 which rotates the
// image such that the image origin is in the top right of the display with
// respect to the raster origin, \b DISPLAY_ROTATE_180 which places the image
// origin in the bottom right of the raster scan or \b DISPLAY_ROTATE_270
// which places the origin at the bottom left of the raster.  In the
// \b DISPLAY_ROTATE_90 and \b DISPLAY_ROTATE_270 cases, the width and height
// reported to the graphics library in the \b g_sGrRaster4BppDriver structure
// are reversed with respect to \b RASTER_WIDTH and \b RASTER_HEIGHT.
//
// This driver assumes that the LCD controller has been configured to use
// palettized pixel formats compatible with GrLib's description.  To ensure
// this, LCDRasterConfigSet() must have been called with flags \b
// RASTER_READ_ORDER_REVERSED and \b RASTER_NIBBLE_MODE_ENABLED set.  If one
// or both of these flags are absent, the display will be corrupted.
//
// \return None
//
//*****************************************************************************
void
GrRaster4BppDriverInit(uint32_t *pui32FrameBuffer)
{
    uint16_t ui16Temp;

    //
    // Fill in all the instance data we need from the passed parameters.
    //
    g_Raster4bppInst.pui16Palette = (uint16_t *)pui32FrameBuffer;
    g_Raster4bppInst.pui8FrameBuffer = (uint8_t *)pui32FrameBuffer + (16 * 2);

    //
    // Initialize the pixel format in the frame buffer header.
    //
    ui16Temp = READ_HWORD(g_Raster4bppInst.pui16Palette);
    ui16Temp &= ~FB_TYPE_MASK;
    ui16Temp |= FB_TYPE_4BPP;
    WRITE_WORD((uint32_t *)g_Raster4bppInst.pui16Palette, ui16Temp);
}

//*****************************************************************************
//
// Draws a pixel on the screen.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
// \param lX is the X coordinate of the pixel.
// \param lY is the Y coordinate of the pixel.
// \param ui32Value is the color index of the pixel.
//
// This function sets the given pixel to a particular color.  The coordinates
// of the pixel are assumed to be within the extents of the display.
//
// \return None.
//
//*****************************************************************************
static void
GrRaster4BppDriverPixelDraw(void *pvDisplayData, int32_t lX, int32_t lY,
                           uint32_t ui32Value)
{
    int32_t i32XMapped, i32YMapped;
    uint8_t *pui8PixByte;
    tRaster4bppDriverInst *pInst;
    uint8_t ui8Pixel;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

    //
    // Set the X address of the display cursor.
    //
    i32XMapped = MAPPED_X(lX, lY);
    i32YMapped = MAPPED_Y(lX, lY);

    //
    // Get a pointer to the byte containing the pixel we need to write.
    //
    pui8PixByte = (uint8_t *)(pInst->pui8FrameBuffer +
                 ((i32YMapped * RASTER_WIDTH) / 2) +
                 (i32XMapped / 2));
    ui8Pixel = *pui8PixByte;

    //
    // Which nibble of this byte represents the required pixel?
    //
    if(i32XMapped & 1)
    {
        WRITE_BYTE(pui8PixByte, (ui8Pixel & 0xF0) | (ui32Value & 0x0F));
    }
    else
    {
        WRITE_BYTE(pui8PixByte, (ui8Pixel & 0x0F) | ((ui32Value & 0x0F) << 4));
    }
}

//*****************************************************************************
//
// Draws a horizontal sequence of pixels on the screen.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
// \param lX is the X coordinate of the first pixel.
// \param lY is the Y coordinate of the first pixel.
// \param lX0 is sub-pixel offset within the pixel data, which is valid for 1
// or 4 bit per pixel formats.
// \param lCount is the number of pixels to draw.
// \param lBPP is the number of bits per pixel ORed with a flag indicating
// whether or not this run represents the start of a new image.
// \param pui8Data is a pointer to the pixel data.  For 1 and 4 bit per pixel
// formats, the most significant bit(s) represent the left-most pixel.
// \param pui8Palette is a pointer to the palette used to draw the pixels.
//
// This function draws a horizontal sequence of pixels on the screen, using
// the supplied palette.  For 1 bit per pixel format, the palette contains
// pre-translated colors; for 4 and 8 bit per pixel formats, the palette
// contains 24-bit RGB values that must be translated before being written to
// the display.
//
// The \e lBPP parameter will take the value 1, 4 or 8 and may be ORed with
// \b GRLIB_DRIVER_FLAG_NEW_IMAGE to indicate that this run represents the
// start of a new image.  Drivers which make use of lookup tables to convert
// from the source to destination pixel values should rebuild their lookup
// table when \b GRLIB_DRIVER_FLAG_NEW_IMAGE is set.
//
// \return None.
//
//*****************************************************************************
#ifdef NORMAL
//
// This version of the function handles the cases where we are writing
// lines of pixels in the same orientation as the raster scan.
//
static void
GrRaster4BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    unsigned char *pui8Ptr;
    uint32_t ui32Byte;
    tRaster4bppDriverInst *pInst;
    uint8_t ui8Byte;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui8Ptr = pInst->pui8FrameBuffer + ((RASTER_WIDTH / 2) * lY) + (lX / 2);

    //
    // Determine the nibble position of the starting pixel.
    //
    lX &= 1;

    //
    // Determine how to interpret the pixel data based on the number of bits
    // per pixel.
    //
    switch(lBPP & 0xFF)
    {
        //
        // The pixel data is in 1 bit per pixel format.
        //
        case 1:
        {
            //
            // Loop while there are more pixels to draw.
            //
            while(lCount)
            {
                //
                // Get the next byte of image data.
                //
                ui32Byte = *pui8Data++;

                //
                // Loop through the pixels in this byte of image data.
                //
                for(; (lX0 < 8) && lCount; lX0++, lCount--)
                {
                    //
                    // Draw this pixel in the appropriate color.
                    //
                    ui8Byte = READ_BYTE(pui8Ptr);
                    WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0xF0 : 0x0F))) |
                              ((((uint32_t *)pui8Palette)[
                               (ui32Byte >> (7 - lX0)) & 1] <<  (lX ? 0 : 4))));

                    lX = 1 - lX;
                    if(!lX)
                    {
                        pui8Ptr++;
                    }
                }

                //
                // Start at the beginning of the next byte of image data.
                //
                lX0 = 0;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 4 bit per pixel format.
        //
        case 4:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 16);
            }

            //
            // Loop while there are more pixels to draw.  "Duff's device" is
            // used to jump into the middle of the loop if the first nibble of
            // the pixel data should not be used.  Duff's device makes use of
            // the fact that a case statement is legal anywhere within a
            // sub-block of a switch statement.  See
            // http://en.wikipedia.org/wiki/Duff's_device for detailed
            // information about Duff's device.
            //
            switch(lX0 & 1)
            {
                case 0:
                    while(lCount)
                    {
                        //
                        // Get the upper nibble of the next byte of pixel data
                        // and extract the corresponding entry from the
                        // palette.
                        //
                        ui32Byte = (*pui8Data >> 4);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        ui8Byte = READ_BYTE(pui8Ptr);
                        WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0xF0 : 0x0F)) |
                                   pInst->pui8Lookup[ui32Byte] << (lX ? 0 : 4)));

                        lX = 1 - lX;
                        if(!lX)
                        {
                            pui8Ptr++;
                        }

                        //
                        // Decrement the count of pixels to draw.
                        //
                        lCount--;

                        //
                        // See if there is another pixel to draw.
                        //
                        if(lCount)
                        {
                case 1:
                            //
                            // Get the lower nibble of the next byte of pixel
                            // data and extract the corresponding entry from
                            // the palette.
                            //
                            ui32Byte = (*pui8Data++ & 15);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            ui8Byte = READ_BYTE(pui8Ptr);
                            WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0xF0 : 0x0F)) |
                                       pInst->pui8Lookup[ui32Byte] <<
                                       (lX ? 0 : 4)));

                            lX = 1 - lX;
                            if(!lX)
                            {
                                pui8Ptr++;
                            }


                            //
                            // Decrement the count of pixels to draw.
                            //
                            lCount--;
                        }
                    }
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 8 bit per pixel format.
        //
        case 8:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 256);
            }

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                ui8Byte = READ_BYTE(pui8Ptr);
                WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0xF0 : 0x0F)) |
                           (pInst->pui8Lookup[ui32Byte] << (lX ? 0 : 4))));

                lX = 1 - lX;
                if(!lX)
                {
                    pui8Ptr++;
                }
            }

            //
            // The image data has been drawn.
            //
            break;
        }
    }
}
#else
#ifdef DISPLAY_ROTATE_180
//
// This version of the function handles the cases where we are writing
// lines of pixels horizontally but in the opposite direction to the raster
// scan.
//
static void
GrRaster4BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    unsigned char *pui8Ptr;
    uint32_t ui32Byte;
    tRaster4bppDriverInst *pInst;
    int32_t lXMapped, lYMapped;
    uint8_t ui8Byte;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

    //
    // Transform the starting point into raster coordinate space.
    //
    lXMapped = MAPPED_X(lX, lY);
    lYMapped = MAPPED_Y(lX, lY);

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui8Ptr = pInst->pui8FrameBuffer +
             ((RASTER_WIDTH / 2) * lYMapped) +
             (lXMapped / 2);

    //
    // Determine the nibble position of the starting pixel.
    //
    lX = 1 - (lXMapped & 1);

    //
    // Determine how to interpret the pixel data based on the number of bits
    // per pixel.
    //
    switch(lBPP & 0xFF)
    {
        //
        // The pixel data is in 1 bit per pixel format.
        //
        case 1:
        {
            //
            // Loop while there are more pixels to draw.
            //
            while(lCount)
            {
                //
                // Get the next byte of image data.
                //
                ui32Byte = *pui8Data++;

                //
                // Loop through the pixels in this byte of image data.
                //
                for(; (lX0 < 8) && lCount; lX0++, lCount--)
                {
                    //
                    // Draw this pixel in the appropriate color.
                    //
                    ui8Byte = READ_BYTE(pui8Ptr);
                    WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0x0F : 0xF0))) |
                              ((((uint32_t *)pui8Palette)[
                               (ui32Byte >> (7 - lX0)) & 1] <<  (lX ? 4 : 0))));

                    lX = 1 - lX;
                    if(!lX)
                    {
                        pui8Ptr--;
                    }
                }

                //
                // Start at the beginning of the next byte of image data.
                //
                lX0 = 0;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 4 bit per pixel format.
        //
        case 4:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 16);
            }

            //
            // Loop while there are more pixels to draw.  "Duff's device" is
            // used to jump into the middle of the loop if the first nibble of
            // the pixel data should not be used.  Duff's device makes use of
            // the fact that a case statement is legal anywhere within a
            // sub-block of a switch statement.  See
            // http://en.wikipedia.org/wiki/Duff's_device for detailed
            // information about Duff's device.
            //
            switch(lX0 & 1)
            {
                case 0:
                    while(lCount)
                    {
                        //
                        // Get the upper nibble of the next byte of pixel data
                        // and extract the corresponding entry from the
                        // palette.
                        //
                        ui32Byte = (*pui8Data >> 4);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        ui8Byte = READ_BYTE(pui8Ptr);
                        WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0x0F : 0xF0)) |
                                   pInst->pui8Lookup[ui32Byte] << (lX ? 4 : 0)));

                        lX = 1 - lX;
                        if(!lX)
                        {
                            pui8Ptr--;
                        }

                        //
                        // Decrement the count of pixels to draw.
                        //
                        lCount--;

                        //
                        // See if there is another pixel to draw.
                        //
                        if(lCount)
                        {
                case 1:
                            //
                            // Get the lower nibble of the next byte of pixel
                            // data and extract the corresponding entry from
                            // the palette.
                            //
                            ui32Byte = (*pui8Data++ & 15);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            ui8Byte = READ_BYTE(pui8Ptr);
                            WRITE_BYTE(pui8Ptr,
                                       ((ui8Byte & (lX ? 0x0F : 0xF0)) |
                                        pInst->pui8Lookup[ui32Byte] <<
                                        (lX ? 4 : 0)));

                            lX = 1 - lX;
                            if(!lX)
                            {
                                pui8Ptr--;
                            }


                            //
                            // Decrement the count of pixels to draw.
                            //
                            lCount--;
                        }
                    }
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 8 bit per pixel format.
        //
        case 8:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 256);
            }

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                ui8Byte = READ_BYTE(pui8Ptr);
                WRITE_BYTE(pui8Ptr, ((ui8Byte & (lX ? 0x0F : 0xF0)) |
                           (pInst->pui8Lookup[ui32Byte] << (lX ? 4 : 0))));

                lX = 1 - lX;
                if(!lX)
                {
                    pui8Ptr--;
                }
            }

            //
            // The image data has been drawn.
            //
            break;
        }
    }
}
#else
#ifdef DISPLAY_ROTATE_90
//
// This version of the function handles the cases where we are writing
// lines of pixels vertically downwards relative to the raster scan.
//
static void
GrRaster4BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    unsigned char *pui8Ptr;
    unsigned char ucMask, ucShift;
    uint32_t ui32Byte;
    tRaster4bppDriverInst *pInst;
    int32_t lXMapped, lYMapped;
    uint8_t ui8Byte;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

    //
    // Transform the starting point into raster coordinate space.
    //
    lXMapped = MAPPED_X(lX, lY);
    lYMapped = MAPPED_Y(lX, lY);

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui8Ptr = pInst->pui8FrameBuffer +
             ((RASTER_WIDTH / 2) * lYMapped) +
             (lXMapped / 2);

    //
    // Determine the nibble position we will be writing.
    //
    lX = lXMapped & 1;
    ucMask = (lX ? 0xF0 : 0x0F);
    ucShift = (lX ? 0 : 4);

    //
    // Determine how to interpret the pixel data based on the number of bits
    // per pixel.
    //
    switch(lBPP & 0xFF)
    {
        //
        // The pixel data is in 1 bit per pixel format.
        //
        case 1:
        {
            //
            // Loop while there are more pixels to draw.
            //
            while(lCount)
            {
                //
                // Get the next byte of image data.
                //
                ui32Byte = *pui8Data++;

                //
                // Loop through the pixels in this byte of image data.
                //
                for(; (lX0 < 8) && lCount; lX0++, lCount--)
                {
                    //
                    // Draw this pixel in the appropriate color.
                    //
                    ui8Byte = READ_BYTE(pui8Ptr);
                    WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask)) |
                              ((((uint32_t *)pui8Palette)[
                               (ui32Byte >> (7 - lX0)) & 1] <<  ucShift)));

                    pui8Ptr += (RASTER_WIDTH / 2);
                }

                //
                // Start at the beginning of the next byte of image data.
                //
                lX0 = 0;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 4 bit per pixel format.
        //
        case 4:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 16);
            }

            //
            // Loop while there are more pixels to draw.  "Duff's device" is
            // used to jump into the middle of the loop if the first nibble of
            // the pixel data should not be used.  Duff's device makes use of
            // the fact that a case statement is legal anywhere within a
            // sub-block of a switch statement.  See
            // http://en.wikipedia.org/wiki/Duff's_device for detailed
            // information about Duff's device.
            //
            switch(lX0 & 1)
            {
                case 0:
                    while(lCount)
                    {
                        //
                        // Get the upper nibble of the next byte of pixel data
                        // and extract the corresponding entry from the
                        // palette.
                        //
                        ui32Byte = (*pui8Data >> 4);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        ui8Byte = READ_BYTE(pui8Ptr);
                        WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask) |
                                   pInst->pui8Lookup[ui32Byte] << ucShift));

                        pui8Ptr += (RASTER_WIDTH / 2);

                        //
                        // Decrement the count of pixels to draw.
                        //
                        lCount--;

                        //
                        // See if there is another pixel to draw.
                        //
                        if(lCount)
                        {
                case 1:
                            //
                            // Get the lower nibble of the next byte of pixel
                            // data and extract the corresponding entry from
                            // the palette.
                            //
                            ui32Byte = (*pui8Data++ & 15);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            ui8Byte = READ_BYTE(pui8Ptr);
                            WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask) |
                                       pInst->pui8Lookup[ui32Byte] << ucShift));

                            pui8Ptr += (RASTER_WIDTH / 2);

                            //
                            // Decrement the count of pixels to draw.
                            //
                            lCount--;
                        }
                    }
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 8 bit per pixel format.
        //
        case 8:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 256);
            }

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                ui8Byte = READ_BYTE(pui8Ptr);
                WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask) |
                           (pInst->pui8Lookup[ui32Byte] << ucShift)));

                pui8Ptr += (RASTER_WIDTH / 2);
            }

            //
            // The image data has been drawn.
            //
            break;
        }
    }
}
#else
#ifdef DISPLAY_ROTATE_270
//
// This version of the function handles the cases where we are writing
// lines of pixels vertically upwards relative to the raster scan.
//
static void
GrRaster4BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    unsigned char *pui8Ptr;
    unsigned char ucMask, ucShift;
    uint32_t ui32Byte;
    tRaster4bppDriverInst *pInst;
    int32_t lXMapped, lYMapped;
    uint8_t ui8Byte;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

    //
    // Transform the starting point into raster coordinate space.
    //
    lXMapped = MAPPED_X(lX, lY);
    lYMapped = MAPPED_Y(lX, lY);

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui8Ptr = pInst->pui8FrameBuffer +
             ((RASTER_WIDTH / 2) * lYMapped) +
             (lXMapped / 2);

    //
    // Determine the nibble position we will be writing.
    //
    lX = lXMapped & 1;
    ucMask = (lX ? 0xF0 : 0x0F);
    ucShift = (lX ? 0 : 4);

    //
    // Determine how to interpret the pixel data based on the number of bits
    // per pixel.
    //
    switch(lBPP & 0xFF)
    {
        //
        // The pixel data is in 1 bit per pixel format.
        //
        case 1:
        {
            //
            // Loop while there are more pixels to draw.
            //
            while(lCount)
            {
                //
                // Get the next byte of image data.
                //
                ui32Byte = *pui8Data++;

                //
                // Loop through the pixels in this byte of image data.
                //
                for(; (lX0 < 8) && lCount; lX0++, lCount--)
                {
                    //
                    // Draw this pixel in the appropriate color.
                    //
                    ui8Byte = READ_BYTE(pui8Ptr);
                    WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask)) |
                              ((((uint32_t *)pui8Palette)[
                               (ui32Byte >> (7 - lX0)) & 1] <<  ucShift)));

                    pui8Ptr -= (RASTER_WIDTH / 2);
                }

                //
                // Start at the beginning of the next byte of image data.
                //
                lX0 = 0;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 4 bit per pixel format.
        //
        case 4:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 16);
            }

            //
            // Loop while there are more pixels to draw.  "Duff's device" is
            // used to jump into the middle of the loop if the first nibble of
            // the pixel data should not be used.  Duff's device makes use of
            // the fact that a case statement is legal anywhere within a
            // sub-block of a switch statement.  See
            // http://en.wikipedia.org/wiki/Duff's_device for detailed
            // information about Duff's device.
            //
            switch(lX0 & 1)
            {
                case 0:
                    while(lCount)
                    {
                        //
                        // Get the upper nibble of the next byte of pixel data
                        // and extract the corresponding entry from the
                        // palette.
                        //
                        ui32Byte = (*pui8Data >> 4);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        ui8Byte = READ_BYTE(pui8Ptr);
                        WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask) |
                                   pInst->pui8Lookup[ui32Byte] << ucShift));

                        pui8Ptr -= (RASTER_WIDTH / 2);

                        //
                        // Decrement the count of pixels to draw.
                        //
                        lCount--;

                        //
                        // See if there is another pixel to draw.
                        //
                        if(lCount)
                        {
                case 1:
                            //
                            // Get the lower nibble of the next byte of pixel
                            // data and extract the corresponding entry from
                            // the palette.
                            //
                            ui32Byte = (*pui8Data++ & 15);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            ui8Byte = READ_BYTE(pui8Ptr);
                            WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask) |
                                       pInst->pui8Lookup[ui32Byte] << ucShift));

                            pui8Ptr -= (RASTER_WIDTH / 2);

                            //
                            // Decrement the count of pixels to draw.
                            //
                            lCount--;
                        }
                    }
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 8 bit per pixel format.
        //
        case 8:
        {
            //
            // Create a lookup table that translates from the source palette
            // into the current frame buffer palette if this is the first line
            // of a new image.
            //
            if(lBPP & GRLIB_DRIVER_FLAG_NEW_IMAGE)
            {
                GenerateImagePaletteLookup(pInst, pui8Palette, 256);
            }

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                ui8Byte = READ_BYTE(pui8Ptr);
                WRITE_BYTE(pui8Ptr, ((ui8Byte & ucMask) |
                           (pInst->pui8Lookup[ui32Byte] << ucShift)));

                pui8Ptr -= (RASTER_WIDTH / 2);
            }

            //
            // The image data has been drawn.
            //
            break;
        }
    }
}
#endif
#endif
#endif
#endif

//*****************************************************************************
//
// Draws a horizontal line relative to the raster origin.  This function
// assumes that coordinates have already been mapped to account for the display
// orientation.
//
//*****************************************************************************
static void
LineDrawHInternal(tRaster4bppDriverInst *pInst, int32_t i32X1, int32_t i32X2,
                  int32_t i32Y,  uint32_t ui32Value)
{
    uint8_t *pui8Data;
    uint8_t ui8Byte;

    //
    // Get a pointer to the frame buffer.
    //
    pui8Data = pInst->pui8FrameBuffer;

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui8Data += ((RASTER_WIDTH / 2) * i32Y) + (i32X1 / 2);

    //
    // Replicate the pixel value into all 8 positions within the word.
    //
    ui32Value &= 0x0F;
    ui32Value |= (ui32Value << 4);
    ui32Value |= (ui32Value << 8);
    ui32Value |= (ui32Value << 16);

    //
    // See if the current buffer byte contains pixels that should be left
    // unmodified.
    //
    if(i32X1 & 1)
    {
        //
        // Draw the appropriate pixels within this byte.
        //
        ui8Byte = READ_BYTE(pui8Data);
        WRITE_BYTE(pui8Data, (ui8Byte & 0xF0) | (ui32Value & 0x0F));
        pui8Data++;
        i32X1++;
    }

    //
    // See if the buffer pointer is not half-word aligned and there are at
    // least two pixels left to draw.
    //
    if(((uint32_t)pui8Data & 1) && ((i32X2 - i32X1) >= 2))
    {
        //
        // Draw two pixels to half-word align the buffer pointer.
        //
        WRITE_BYTE(pui8Data, (ui32Value & 0xff));
        pui8Data++;
        i32X1 += 2;
    }

    //
    // See if the buffer pointer is not word aligned and there are at least
    // four pixels left to draw.
    //
    if(((uint32_t)pui8Data & 2) && ((i32X2 - i32X1) >= 4))
    {
        //
        // Draw four pixels to word align the buffer pointer.
        //
        WRITE_HWORD((uint16_t *)pui8Data, (ui32Value & 0xffff));
        pui8Data += 2;
        i32X1 += 4;
    }

    //
    // Loop while there are at least 8 pixels left to draw.
    //
    while((i32X1 + 7) <= i32X2)
    {
        //
        // Draw 8 pixels.
        //
        WRITE_WORD((uint32_t *)pui8Data, ui32Value);
        pui8Data += 4;
        i32X1 += 8;
    }

    //
    // See if there are at least four pixels left to draw.
    //
    if((i32X1 + 3) <= i32X2)
    {
        //
        // Draw 4 pixels, leaving the buffer pointer half-word aligned.
        //
        WRITE_HWORD((uint16_t *)pui8Data, (ui32Value & 0xffff));
        pui8Data += 2;
        i32X1 += 4;
    }

    //
    // See if there are at least two pixels left to draw.
    //
    if((i32X1 + 1) <= i32X2)
    {
        //
        // Draw two pixels, leaving the buffer pointer byte aligned.
        //
        WRITE_BYTE(pui8Data, (ui32Value & 0xff));
        pui8Data++;
        i32X1 += 2;
    }

    //
    // See if there are any pixels left to draw.
    //
    if(i32X1 <= i32X2)
    {
        //
        // Draw the remaining pixel.
        //
        ui8Byte = READ_BYTE(pui8Data);
        WRITE_BYTE(pui8Data, (ui8Byte & 0x0F) | (ui32Value & 0xF0));
    }
}

//*****************************************************************************
//
// Draws a vertical line relative to the raster origin.  This function assumes
// that coordinates have already been mapped to account for the display
// orientation.
//
//*****************************************************************************
static void
LineDrawVInternal(tRaster4bppDriverInst *pInst, int32_t i32X, int32_t i32Y1, int32_t i32Y2,
                  uint32_t ui32Value)
{
    uint8_t *pui8PixelByte;
    uint8_t ui8BitMask, ui8Value;
    int32_t lRows;
    uint8_t ui8Byte;

    //
    // Get a pointer to the byte containing the pixel we need to write.
    //
    pui8PixelByte = (uint8_t *)(pInst->pui8FrameBuffer +
                             (i32Y1 * (RASTER_WIDTH / 2)) + (i32X / 2));

    //
    // Which nibble of this byte represents the required pixel?
    //
    if(i32X & 1)
    {
        ui8BitMask = 0xF0;
        ui8Value = (uint8_t)(ui32Value & 0x0F);
    }
    else
    {
        ui8BitMask = 0x0F;
        ui8Value = (uint8_t)((ui32Value & 0x0F) << 4);
    }

    //
    // Set the required pixels on each row to draw the line.
    //
    for(lRows = i32Y1; lRows <= i32Y2; lRows++)
    {
        ui8Byte = READ_BYTE(pui8PixelByte);
        WRITE_BYTE(pui8PixelByte, (ui8Byte & ui8BitMask) | ui8Value);
        pui8PixelByte += (RASTER_WIDTH / 2);
    }
}

//*****************************************************************************
//
// Draws a horizontal line.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
// \param lX1 is the X coordinate of the start of the line.
// \param lX2 is the X coordinate of the end of the line.
// \param lY is the Y coordinate of the line.
// \param ui32Value is the color of the line.
//
// This function draws a horizontal line on the display.  The coordinates of
// the line are assumed to be within the extents of the display.
//
// \return None.
//
//*****************************************************************************
static void
GrRaster4BppDriverLineDrawH(void *pvDisplayData, int32_t lX1, int32_t lX2,
                            int32_t lY, uint32_t ui32Value)
{
    tRaster4bppDriverInst *pInst;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

#ifdef NORMAL
    //
    // Normal display orientation case.
    //
    LineDrawHInternal(pInst, lX1, lX2, lY, ui32Value);
#else
#ifdef DISPLAY_ROTATE_180
    //
    // 180 degree rotated case.  Horizontal lines are still horizontal but
    // the sorted order of the Xs reverses.
    //
    LineDrawHInternal(pInst, MAPPED_X(lX2, lY), MAPPED_X(lX1, lY),
                      MAPPED_Y(lX1, lY), ui32Value);
#else
#ifdef DISPLAY_ROTATE_90
    //
    // 90 degree rotated case. Horizontal lines are now vertical with respect
    // to the scan.
    //
    LineDrawVInternal(pInst, MAPPED_X(lX2, lY), MAPPED_Y(lX1, lY),
                      MAPPED_Y(lX2, lY), ui32Value);
#else
#ifdef DISPLAY_ROTATE_270
    //
    // 90 degree rotated case. Horizontal lines are now vertical with respect
    // to the scan and the Ys reverse.
    //
    LineDrawVInternal(pInst, MAPPED_X(lX2, lY), MAPPED_Y(lX2, lY),
                      MAPPED_Y(lX1, lY), ui32Value);
#endif
#endif
#endif
#endif
}

//*****************************************************************************
//
// Draws a vertical line.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
// \param lX is the X coordinate of the line.
// \param lY1 is the Y coordinate of the start of the line.
// \param lY2 is the Y coordinate of the end of the line.
// \param ui32Value is the color of the line.
//
// This function draws a vertical line on the display.  The coordinates of the
// line are assumed to be within the extents of the display.
//
// \return None.
//
//*****************************************************************************
static void
GrRaster4BppDriverLineDrawV(void *pvDisplayData, int32_t lX, int32_t lY1,
                            int32_t lY2, uint32_t ui32Value)
{
    tRaster4bppDriverInst *pInst;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

#ifdef NORMAL
    //
    // Normal display orientation case.
    //
    LineDrawVInternal(pInst, lX, lY1, lY2, (uint32_t)ui32Value);
#else
#ifdef DISPLAY_ROTATE_180
    //
    // 180 degree rotated case.  Vertical lines are still vertical but the
    // sorted order of the Ys reverses.
    //
    LineDrawVInternal(pInst, MAPPED_X(lX, lY1), MAPPED_Y(lX, lY2),
                      MAPPED_Y(lX, lY1), (uint32_t)ui32Value);
#else
#ifdef DISPLAY_ROTATE_90
    //
    // 90 degree rotated case. Vertical lines are now horizontal with respect
    // to the scan.
    //
    LineDrawHInternal(pInst, MAPPED_X(lX, lY2), MAPPED_X(lX, lY1),
                      MAPPED_Y(lX, lY1), (uint32_t)ui32Value);
#else
#ifdef DISPLAY_ROTATE_270
    //
    // 90 degree rotated case. Horizontal lines are now vertical with respect
    // to the scan and the Ys reverse.
    //
    LineDrawHInternal(pInst, MAPPED_X(lX, lY1), MAPPED_X(lX, lY2),
                      MAPPED_Y(lX, lY2), (uint32_t)ui32Value);
#endif
#endif
#endif
#endif
}

//*****************************************************************************
//
// Fills a rectangle.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
// \param pRect is a pointer to the structure describing the rectangle.
// \param ui32Value is the color of the rectangle.
//
// This function fills a rectangle on the display.  The coordinates of the
// rectangle are assumed to be within the extents of the display, and the
// rectangle specification is fully inclusive (in other words, both sXMin and
// sXMax are drawn, along with sYMin and sYMax).
//
// \return None.
//
//*****************************************************************************
static void
GrRaster4BppDriverRectFill(void *pvDisplayData, const tRectangle *pRect,
                           uint32_t ui32Value)
{
    int32_t lTop, lBottom, lLeft, lRight;
    tRaster4bppDriverInst *pInst;
    int32_t lLine;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)pvDisplayData;

    //
    // Map the coordinates given (in rotated coordinate space) back into the
    // raster coordinate space.
    //
#ifdef NORMAL
    lTop = (int32_t)pRect->i16YMin;
    lBottom = (int32_t)pRect->i16YMax;
    lLeft = (int32_t)pRect->i16XMin;
    lRight = (int32_t)pRect->i16XMax;
#else
#ifdef DISPLAY_ROTATE_180
    lTop = (RASTER_HEIGHT - 1) - (int32_t)pRect->i16YMax;
    lBottom = (RASTER_HEIGHT - 1) - (int32_t)pRect->i16YMin;
    lLeft = (RASTER_WIDTH - 1) - (int32_t)pRect->i16XMax;
    lRight = (RASTER_WIDTH - 1) - (int32_t)pRect->i16XMin;
#else
#ifdef DISPLAY_ROTATE_90
    lTop = (int32_t)pRect->i16XMin;
    lBottom = (int32_t)pRect->i16XMax;
    lLeft = (RASTER_WIDTH - 1) - (int32_t)pRect->i16YMax;
    lRight = (RASTER_WIDTH - 1) - (int32_t)pRect->i16YMin;
#else
#ifdef DISPLAY_ROTATE_270
    lTop = (RASTER_HEIGHT - 1) - (int32_t)pRect->i16XMax;
    lBottom = (RASTER_HEIGHT - 1) - (int32_t)pRect->i16XMin;
    lLeft = (int32_t)pRect->i16YMin;
    lRight = (int32_t)pRect->i16YMax;
#endif
#endif
#endif
#endif

    //
    // Draw horizontal lines to fill the rectangular region.  Note that we've
    // remapped the coordinates passed into raster-origin values already.
    //
    for(lLine = lTop; lLine <= lBottom; lLine++)
    {
        LineDrawHInternal(pInst, lLeft, lRight, lLine, ui32Value);
    }
}

//*****************************************************************************
//
// Translates a 24-bit RGB color to a 4bpp driver-specific color.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
// \param ui32Value is the 24-bit RGB color.  The least-significant byte is the
// blue channel, the next byte is the green channel, and the third byte is the
// red channel.
//
// This function translates a 24-bit RGB color into a value that can be
// written into the display's 4bpp frame buffer in order to reproduce that
// color, or the closest possible approximation of that color.
//
// Given that applications using indexed color frame buffers generally work
// with palette indices rather than 24-bit RGB colors, this function merely
// returns the bottom 4 bits of the color passed, assuming that the original
// color is an index rather than an RGB value.
//
// \return Returns the display-driver specific color.
//
//*****************************************************************************
static uint32_t
GrRaster4BppDriverColorTranslate(void *pvDisplayData,  uint32_t ui32Value)
{
    ASSERT(pvDisplayData);

    //
    // Translate from a 24-bit RGB color to a 4 bit color index.
    //
    return(DPYCOLORTRANSLATE(pvDisplayData, ui32Value));
}

//*****************************************************************************
//
// Flushes any cached drawing operations.
//
// \param pvDisplayData is a pointer to the driver-specific data for this
// display driver.
//
// This functions flushes any cached drawing operations to the display.  This
// is useful when a local frame buffer is used for drawing operations, and the
// flush would copy the local frame buffer to the display.  For the SSD2119
// driver, the flush is a no operation.
//
// \return None.
//
//*****************************************************************************
static void
GrRaster4BppDriverFlush(void *pvDisplayData)
{
    //
    // There is nothing to be done.
    //
}

//*****************************************************************************
//
// Sets the color palette for the frame buffer image.
//
// \param psDisplay points to the structure defining the display driver
// for which the palette is to be set.
// \param pui32Palette points to memory containing an array of 24-bit RGB
// colors to use in setting the palette.
// \param ui32Offset is the index of the first palette entry to set.  This may
// be any value between 0 and 15 for the 4bpp driver.
// \param ui32Count is the number of colors in the \e pui32Palette table that are
// to be written into the palette.
//
// This function sets the color palette for the display driver and controls
// the colors associated with pixel values 0 through 15.
//
// \return None.
//
//*****************************************************************************
void
GrRaster4BppPaletteSet(tDisplay *psDisplay, uint32_t *pui32Palette,
                       uint32_t ui32Offset, uint32_t ui32Count)
{
    uint32_t ui32Loop;
    uint16_t ui16Entry, ui16Last;
    tRaster4bppDriverInst *pInst;

    ASSERT(ui32Offset <= 15);
    ASSERT(pui32Palette);
    ASSERT(ui32Count <= 16);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster4bppDriverInst *)(psDisplay->pvDisplayData);

    //
    // Step through the palette entries we are to set.
    //
    for(ui32Loop = 0; ui32Loop < ui32Count; ui32Loop++)
    {
        //
        // Get the existing palette entry.
        //
        ui16Last = READ_HWORD(&pInst->pui16Palette[ui32Offset + ui32Loop]);

        //
        // Convert the RGB888 color from GrLib into the hardware's RGB444
        // palette format.
        //
        ui16Entry = GRLIB_COLOR_TO_PAL_ENTRY(pui32Palette[ui32Loop]);

        //
        // Write the new color to the palette, preserving the top nibble.
        //
        ui16Entry |= (ui16Last & 0xF000);
        WRITE_HWORD(&pInst->pui16Palette[ui32Offset + ui32Loop], ui16Entry);
    }
}

//*****************************************************************************
//
// The display structure that describes the driver when used to target a
// 4bpp, 16 color frame buffer.
//
//*****************************************************************************
tDisplay g_sGrRaster4BppDriver =
{
    sizeof(tDisplay),
    &g_Raster4bppInst,
#if (defined NORMAL) || (defined DISPLAY_ROTATE_180)
    RASTER_WIDTH,
    RASTER_HEIGHT,
#else
    RASTER_HEIGHT,
    RASTER_WIDTH,
#endif
    GrRaster4BppDriverPixelDraw,
    GrRaster4BppDriverPixelDrawMultiple,
    GrRaster4BppDriverLineDrawH,
    GrRaster4BppDriverLineDrawV,
    GrRaster4BppDriverRectFill,
    GrRaster4BppDriverColorTranslate,
    GrRaster4BppDriverFlush
};
