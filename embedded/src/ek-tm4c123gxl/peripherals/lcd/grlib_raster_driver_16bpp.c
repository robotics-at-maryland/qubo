//*****************************************************************************
//
// grlib_raster_driver_16bpp.c - Generic graphics library display driver
// supporting screens attached to the LCD controller via its raster interface
// and making use of a 16bpp, RGB565 display buffer.
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
#include "drivers/grlib_raster_driver_16bpp.h"

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
#define READ_HWORD(Addr) EPIWorkaroundHWordRead(Addr)
#else
#define WRITE_WORD(Addr, Val)  HWREG(Addr) = (Val)
#define WRITE_HWORD(Addr, Val) HWREGH(Addr) = (Val)
#define READ_HWORD(Addr) HWREGH(Addr)
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
#define FB_TYPE_16BPP       0x4000

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

#define RED_FROM_PAL_ENTRY(x)   (((x) & GRLIB_RED_MASK) >> GRLIB_RED_SHIFT)
#define GREEN_FROM_PAL_ENTRY(x) (((x) & GRLIB_GREEN_MASK) >> GRLIB_GREEN_SHIFT)
#define BLUE_FROM_PAL_ENTRY(x)  (((x) & GRLIB_BLUE_MASK) >> GRLIB_BLUE_SHIFT)

//
// Determine a pixel value from a 24-bit image palette entry.
//
#define PIXEL_FROM_COLOR(c) (((RED_FROM_PAL_ENTRY(c) & 0xF8) << 8) |          \
                             ((GREEN_FROM_PAL_ENTRY(c) & 0xFC) << 3) |        \
                             ((BLUE_FROM_PAL_ENTRY(c) & 0xF8) >> 3))

//*****************************************************************************
//
// Driver instance data structure
//
//*****************************************************************************
typedef struct
{
    uint16_t *pui16FrameBuffer;
    uint16_t *pui16Palette;
    uint8_t  pui8Lookup[256];
}
tRaster16BppDriverInst;

tRaster16BppDriverInst g_Raster16BppInst;

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
// Translates a 24-bit RGB color to a display driver-specific 16bpp color.
//
// \param c is the 24-bit RGB color.  The least-significant byte is the blue
// channel, the next byte is the green channel, and the third byte is the red
// channel.
//
// This macro translates a 24-bit RGB color into a value that can be written
// into the display's 16bpp frame buffer in order to reproduce that color, or
// or closest possible approximation of that color.
//
// \return Returns the display-driver specific color.
//
//*****************************************************************************
#define DPYCOLORTRANSLATE(c)                                                  \
            (((((c) & GRLIB_BLUE_MASK) >> (GRLIB_BLUE_SHIFT + 3))) |          \
             ((((c) & GRLIB_GREEN_MASK) >> (GRLIB_GREEN_SHIFT + 2)) << 5) |   \
             ((((c) & GRLIB_RED_MASK) >> (GRLIB_RED_SHIFT + 3)) << 11))

//*****************************************************************************
//
// Configures the 8bpp raster display driver.
//
// \param pui32FrameBuffer points to the first byte of the frame buffer
// memory.  This corresponds to the first byte of the palette header for
// the frame buffer.
//
// This function must be called to initialize the 1bpp raster display driver
// for use with the Stellaris Graphics Library and the supplied frame buffer.
// The caller is responsible for allocating frame buffer memory and
// configuring the LCD controller appropriately prior to calling this function.
//
// The frame buffer provided must conform to the requirements imposed by the
// LCD controller raster engine. The buffer starts with a header containing
// the color palette and pixel format information. This header is followed
// by the \b RASTER_WIDTH * \b RASTER_HEIGHT pixel array.  The palette
// contains 256 half words for an 8bpp frame buffer.
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
// reported to the graphics library in the \b g_sGrRaster16BppDriver structure
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
GrRaster16BppDriverInit(uint32_t *pui32FrameBuffer)
{
    uint16_t ui16Temp;

    //
    // Fill in all the instance data we need from the passed parameters.
    //
    g_Raster16BppInst.pui16Palette = (uint16_t *)pui32FrameBuffer;
    g_Raster16BppInst.pui16FrameBuffer = (uint16_t *)pui32FrameBuffer + 16;

    //
    // Initialize the pixel format in the frame buffer header.
    //
    ui16Temp = READ_HWORD(g_Raster16BppInst.pui16Palette);
    ui16Temp &= ~FB_TYPE_MASK;
    ui16Temp |= FB_TYPE_16BPP;
    WRITE_HWORD(g_Raster16BppInst.pui16Palette, ui16Temp);
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
GrRaster16BppDriverPixelDraw(void *pvDisplayData, int32_t lX, int32_t lY,
                           uint32_t ui32Value)
{
    int32_t i32XMapped, i32YMapped;
    uint16_t *pui16Pixel;
    tRaster16BppDriverInst *pInst;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

    //
    // Set the X address of the display cursor.
    //
    i32XMapped = MAPPED_X(lX, lY);
    i32YMapped = MAPPED_Y(lX, lY);

    //
    // Get a pointer to the byte containing the pixel we need to write.
    //
    pui16Pixel = pInst->pui16FrameBuffer +
                              (i32YMapped * RASTER_WIDTH) + i32XMapped;

    //
    // Write the pixel.
    //
    WRITE_HWORD(pui16Pixel, (uint16_t)ui32Value);
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
GrRaster16BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    uint16_t *pui16Ptr;
    uint32_t ui32Byte;
    tRaster16BppDriverInst *pInst;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui16Ptr = pInst->pui16FrameBuffer + (RASTER_WIDTH * lY) + lX;

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
            // This case requires a palette.
            //
            ASSERT(pui8Palette);

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
                    WRITE_HWORD(pui16Ptr,
                                 (((uint32_t *)pui8Palette)[(ui32Byte >>
                                                            (7 - lX0)) & 1]));
                    pui16Ptr++;
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
            // This case requires a palette.
            //
            ASSERT(pui8Palette);

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
                        ui32Byte = (*pui8Data >> 4) * 3;
                        ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                    0x00ffffff);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));

                        pui16Ptr++;

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
                            ui32Byte = (*pui8Data & 0x0F) * 3;
                            ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                        0x00ffffff);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));

                            //
                            // Move on to the next source and destination pixel.
                            //
                            pui8Data++;
                            pui16Ptr++;

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
            // This case requires a palette.
            //
            ASSERT(pui8Palette);

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data * 3;
                ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                            0x00ffffff);

                //
                // Translate this palette entry and write it to the screen.
                //
                WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));

                //
                // Move on to the next source and destination pixel.
                //
                pui8Data++;
                pui16Ptr++;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 16 bit per pixel format.
        //
        case 16:
        {
            uint16_t ui16Byte;

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui16Byte = *((uint16_t *)pui8Data);
                pui8Data += 2;

                //
                // Copy this pixel directly to the display buffer.
                //
                WRITE_HWORD(pui16Ptr, ui16Byte);
                pui16Ptr++;
            }
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
GrRaster16BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    uint16_t *pui16Ptr;
    uint32_t ui32Byte;
    tRaster16BppDriverInst *pInst;
    int32_t lXMapped, lYMapped;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

    //
    // Transform the starting point into raster coordinate space.
    //
    lXMapped = MAPPED_X(lX, lY);
    lYMapped = MAPPED_Y(lX, lY);

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui16Ptr = pInst->pui16FrameBuffer + (RASTER_WIDTH * lYMapped) + lXMapped;

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
                    WRITE_HWORD(pui16Ptr,
                                ((uint32_t *)pui8Palette)[(ui32Byte >>
                                                          (7 - lX0)) & 1]);
                    pui16Ptr--;
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
                        ui32Byte = (*pui8Data >> 4) * 3;
                        ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                    0x00ffffff);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));

                        pui16Ptr--;

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
                            ui32Byte = (*pui8Data & 0x0F) * 3;
                            ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                        0x00ffffff);

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));

                            //
                            // Move on to the next source and destination pixel.
                            //
                            pui8Data++;
                            pui16Ptr--;

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
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data * 3;
                ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                            0x00ffffff);
                pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));

                pui16Ptr--;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 16 bit per pixel format.
        //
        case 16:
        {
            uint16_t ui16Byte;

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui16Byte = *((uint16_t *)pui8Data);
                pui8Data += 2;

                //
                // Copy this pixel directly to the display buffer.
                //
                WRITE_HWORD(pui16Ptr, ui16Byte);
                pui16Ptr--;
            }
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
GrRaster16BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    uint16_t *pui16Ptr;
    uint32_t ui32Byte;
    tRaster16BppDriverInst *pInst;
    int32_t lXMapped, lYMapped;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

    //
    // Transform the starting point into raster coordinate space.
    //
    lXMapped = MAPPED_X(lX, lY);
    lYMapped = MAPPED_Y(lX, lY);

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui16Ptr = pInst->pui16FrameBuffer + (RASTER_WIDTH * lYMapped) + lXMapped;

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
                    WRITE_HWORD(pui16Ptr, (((uint32_t *)pui8Palette)[
                                       (ui32Byte >> (7 - lX0)) & 1]));

                    pui16Ptr += RASTER_WIDTH;
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
                        ui32Byte = (*pui8Data >> 4) * 3;
                        ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                    0x00ffffff);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));
                        pui16Ptr += RASTER_WIDTH;

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
                            ui32Byte = (*pui8Data & 0x0F) * 3;
                            ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                        0x00ffffff);
                            pui8Data++;

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));
                            pui16Ptr += RASTER_WIDTH;

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
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data * 3;
                ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                            0x00ffffff);
                pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));
                pui16Ptr += RASTER_WIDTH;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 16 bit per pixel format.
        //
        case 16:
        {
            uint16_t ui16Byte;

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui16Byte = *((uint16_t *)pui8Data);
                pui8Data += 2;

                //
                // Copy this pixel directly to the display buffer.
                //
                WRITE_HWORD(pui16Ptr, ui16Byte);
                pui16Ptr += RASTER_WIDTH;
            }
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
GrRaster16BppDriverPixelDrawMultiple(void *pvDisplayData, int32_t lX,
                                    int32_t lY, int32_t lX0, int32_t lCount,
                                    int32_t lBPP, const uint8_t *pui8Data,
                                    const uint8_t *pui8Palette)
{
    uint16_t *pui16Ptr;
    uint32_t ui32Byte;
    tRaster16BppDriverInst *pInst;
    int32_t lXMapped, lYMapped;

    //
    // Check the arguments.
    //
    ASSERT(pvDisplayData);
    ASSERT(pui8Data);
    ASSERT(pui8Palette);

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

    //
    // Transform the starting point into raster coordinate space.
    //
    lXMapped = MAPPED_X(lX, lY);
    lYMapped = MAPPED_Y(lX, lY);

    //
    // Get the offset to the byte of the image buffer that contains the
    // starting pixel.
    //
    pui16Ptr = pInst->pui16FrameBuffer + (RASTER_WIDTH * lYMapped) + lXMapped;

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
                    WRITE_HWORD(pui16Ptr, (((uint32_t *)pui8Palette)[
                               (ui32Byte >> (7 - lX0)) & 1]));
                    pui16Ptr -= RASTER_WIDTH;
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
                        ui32Byte = (*pui8Data >> 4) * 3;
                        ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                    0x00ffffff);

                        //
                        // Translate this palette entry and write it to the
                        // screen.
                        //
                        WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));
                        pui16Ptr -= RASTER_WIDTH;

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
                            ui32Byte = (*pui8Data & 0x0F) * 3;
                            ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                                        0x00ffffff);
                            pui8Data++;

                            //
                            // Translate this palette entry and write it to the
                            // screen.
                            //
                            WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));
                            pui16Ptr -= RASTER_WIDTH;

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
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui32Byte = *pui8Data * 3;
                ui32Byte = (*(uint32_t *)(pui8Palette + ui32Byte) &
                            0x00ffffff);
                pui8Data++;

                //
                // Translate this palette entry and write it to the screen.
                //
                WRITE_HWORD(pui16Ptr, PIXEL_FROM_COLOR(ui32Byte));
                pui16Ptr -= RASTER_WIDTH;
            }

            //
            // The image data has been drawn.
            //
            break;
        }

        //
        // The pixel data is in 16 bit per pixel format.
        //
        case 16:
        {
            uint16_t ui16Byte;

            //
            // Loop while there are more pixels to draw.
            //
            while(lCount--)
            {
                //
                // Get the next byte of pixel data.
                //
                ui16Byte = *((uint16_t *)pui8Data);
                pui8Data += 2;

                //
                // Copy this pixel directly to the display buffer.
                //
                WRITE_HWORD(pui16Ptr, ui16Byte);
                pui16Ptr -= RASTER_WIDTH;
            }
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
LineDrawHInternal(tRaster16BppDriverInst *pInst, int32_t i32X1, int32_t i32X2,
                  int32_t i32Y,  uint32_t ui32Value)
{
    uint16_t *pui16Data;

    //
    // Get a pointer to the frame buffer.
    //
    pui16Data = pInst->pui16FrameBuffer;

    //
    // Get a pointer to the starting pixel.
    //
    pui16Data += ((RASTER_WIDTH * i32Y) + i32X1);

    //
    // Replicate the pixel value into both positions within the word.
    //
    ui32Value &= 0xFFFF;
    ui32Value |= (ui32Value << 16);

    //
    // See if the buffer pointer is not word aligned and there is at least
    // one pixel left to draw.
    //
    if(((uint32_t)pui16Data & 2) && ((i32X2 - i32X1) > 0))
    {
        //
        // Draw one pixel to word align the buffer pointer.
        //
        WRITE_HWORD(pui16Data, (ui32Value & 0xFFFF));
        pui16Data ++;
        i32X1 += 1;
    }

    //
    // Loop while there are at least 2 pixels left to draw.
    //
    while((i32X1 + 1) <= i32X2)
    {
        //
        // Draw 2 pixels.
        //
        *(uint32_t *)pui16Data = ui32Value;
        WRITE_WORD((uint32_t *)pui16Data, ui32Value);
        pui16Data += 2;
        i32X1 += 2;
    }

    //
    // See if there is a final pixel left to draw.
    //
    if(i32X1 <= i32X2)
    {
        //
        // Draw one pixel,.
        //
        WRITE_HWORD(pui16Data, (ui32Value & 0xFFFF));
        pui16Data++;
        i32X1 += 1;
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
LineDrawVInternal(tRaster16BppDriverInst *pInst, int32_t i32X, int32_t i32Y1, int32_t i32Y2,
                  uint32_t ui32Value)
{
    uint16_t *pui16Pixel;
    int32_t lRows;

    //
    // Get a pointer to the byte containing the pixel we need to write.
    //
    pui16Pixel = pInst->pui16FrameBuffer + (i32Y1 * RASTER_WIDTH) + i32X;

    //
    // Set the required pixels on each row to draw the line.
    //
    for(lRows = i32Y1; lRows <= i32Y2; lRows++)
    {
        WRITE_HWORD(pui16Pixel, (uint16_t)ui32Value);
        pui16Pixel += RASTER_WIDTH;
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
GrRaster16BppDriverLineDrawH(void *pvDisplayData, int32_t lX1, int32_t lX2,
                            int32_t lY, uint32_t ui32Value)
{
    tRaster16BppDriverInst *pInst;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

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
GrRaster16BppDriverLineDrawV(void *pvDisplayData, int32_t lX, int32_t lY1,
                            int32_t lY2, uint32_t ui32Value)
{
    tRaster16BppDriverInst *pInst;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

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
GrRaster16BppDriverRectFill(void *pvDisplayData, const tRectangle *pRect,
                           uint32_t ui32Value)
{
    int32_t lTop, lBottom, lLeft, lRight;
    tRaster16BppDriverInst *pInst;
    int32_t lLine;

    //
    // Get our instance pointer.
    //
    pInst = (tRaster16BppDriverInst *)pvDisplayData;

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
// written into the display's 8bpp frame buffer in order to reproduce that
// color, or the closest possible approximation of that color.  The current
// color palette is scanned for the closest matching color and the index of
// this palette entry is returned.
//
// \return Returns the display-driver specific color.
//
//*****************************************************************************
static uint32_t
GrRaster16BppDriverColorTranslate(void *pvDisplayData,  uint32_t ui32Value)
{
    ASSERT(pvDisplayData);

    //
    // Translate from a 24-bit RGB color to an 8 bit color index.
    //
    return(DPYCOLORTRANSLATE(ui32Value));
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
GrRaster16BppDriverFlush(void *pvDisplayData)
{
    //
    // There is nothing to be done.
    //
}

//*****************************************************************************
//
// The display structure that describes the driver when used to target an
// 8bpp, 256 color frame buffer.
//
//*****************************************************************************
tDisplay g_sGrRaster16BppDriver =
{
    sizeof(tDisplay),
    &g_Raster16BppInst,
#if (defined NORMAL) || (defined DISPLAY_ROTATE_180)
    RASTER_WIDTH,
    RASTER_HEIGHT,
#else
    RASTER_HEIGHT,
    RASTER_WIDTH,
#endif
    GrRaster16BppDriverPixelDraw,
    GrRaster16BppDriverPixelDrawMultiple,
    GrRaster16BppDriverLineDrawH,
    GrRaster16BppDriverLineDrawV,
    GrRaster16BppDriverRectFill,
    GrRaster16BppDriverColorTranslate,
    GrRaster16BppDriverFlush
};
