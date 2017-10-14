//*****************************************************************************
//
// raster_example.c - An example showing the use of the TM4C129x LCD
//                    controller in raster mode.
//
// Copyright (c) 2012-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_sysctl.h"
#include "inc/hw_types.h"
#include "inc/hw_memmap.h"
#include "inc/hw_gpio.h"
#include "inc/hw_ints.h"
#include "inc/hw_lcd.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/lcd.h"
#include "driverlib/interrupt.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "grlib/grlib.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "drivers/grlib_raster_driver_1bpp.h"
#include "drivers/grlib_raster_driver_4bpp.h"
#include "drivers/grlib_raster_driver_8bpp.h"
#include "drivers/grlib_raster_driver_16bpp.h"
#include "drivers/raster_displays.h"
#include "drivers/sdram.h"

//*****************************************************************************
//
//! \addtogroup lcd_examples_list
//! <h1>LCD Controller Raster Mode Example (raster_example)</h1>
//!
//! This application illustrates the use of the Tivaware Graphics Library
//! and Tiva TM4C129x LCD controller driving an 800x480 display using raster
//! (HSYNC/VSYNC/ACTIVE/DATA) mode.  The display is initialized and enabled
//! then a simple pattern including lines, a small image, some text and a
//! circle is displayed.
//!
//! By default, the application is set up to support an Innolux EJ090NA-03A
//! display with 800x480 resolution, refreshed at 60Hz from a 16bpp frame
//! buffer stored in SDRAM.  The SDRAM is attached to the MCU via the External
//! Peripheral Interface (EPI) module.  The file drivers/raster_displays.c
//! contains timings and initialization functions for several other displays
//! and the application can be easily rebuilt to support any of these by
//! replacing the preprocessor define ``\b INNOLUX_DISPLAY'' with one of the
//! other display labels:
//!
//! - \b OPTREX_DISPLAY supports an Optrex T-55226D043J-LW-A-AAN in 800x480
//! with 75Hz refresh rate.
//! - \b LXD_DISPLAY supports an LXD M7170A in 640x480 with 60Hz refresh rate.
//! - \b FORMIKE_DISPLAY supports at Formike KWH070KQ13 in 800x480 with 60Hz
//! refresh rate.
//!
//! Display interface timing information and any required initialization code
//! is included in the file lcd/drivers/raster_displays.c.  New raster-mode
//! displays can be added to this file and raster_displays.h very easily and
//! used by the application merely by adding another display label and
//! appropriate code to set the tRasterDisplayInfo timing structure for that
//! display at the top of raster_example.c.
//!
//! Once appropriate display timings have been determined, the display can be
//! used by the TivaWare Graphics Library via one of the supplied raster mode
//! display drivers.  Four distinct drivers are supplied in the lcd/drivers
//! directory, each supporting a different color depth for the frame buffer:
//!
//! - \b grlib_raster_driver_1bpp.c supports a monochrome (2 color) display
//! buffer.
//! - \b grlib_raster_driver_4bpp.c supports a 4 bit per pixel (16 color) frame
//! buffer.
//! - \b grlib_raster_driver_8bpp supports an 8 bit per pixel (256 color) frame
//! buffer.
//! - \b grlib_raster_driver_16bpp supports a 16 bit per pixel (65536 color)
//! frame buffer.
//!
//! The size of frame buffer required varies with the resolution of the LCD
//! display in use and the desired frame buffer color depth.  Note that the
//! frame buffer color depth may be lower than the native color resolution of
//! the LCD panel - the LCD controller makes use of a color lookup table or
//! palette to convert the pixels in the frame buffer to the correct color
//! format for the LCD's hardware interface.
//!
//! The size of frame buffer, in bytes, can be determined using the following
//! formula:
//!
//! Buffer Size = X * Y * (BPP / 8) + (Header Size)
//!
//! where:
//!
//! - X is the horizontal pixel resolution of the LCD panel
//! - Y is the vertical pixel resolution of the LCD panel
//! - BPP is the desired number of bits per pixel for the frame buffer
//! - Header Size is 512 for 8bpp frame buffers or 32 for all other color
//! resolutions.
//!
//! The frame buffer header contains information informing the LCD controller
//! of the pixel format in the frame buffer and also the color lookup table
//! used for 1, 4 and 8bpp cases.  Note that a 32 byte header is still required
//! even when using 16bpp frame buffers which do not require a color lookup
//! table.
//!
//! For large panels such as those described in raster_displays.h, a frame
//! buffer supporting more than two colors is likely to be too large to fit
//! in the internal memory of a TM4C129x device and would, therefore, require
//! the use of external, EPI-connected SDRAM.  The 16bpp 800x480 frame buffer
//! used in this application requires almost 940KB of RAM for example. For lower
//! resolution displays or lower color depths, internal SRAM may be suitable
//! for use as the frame buffer.  For example, a 16bpp QVGA (320x240) frame
//! buffer occupies about 150KB of storage and a monochrome (1bpp) 800x480
//! frame buffer needs only 48KB.
//
//*****************************************************************************

//*****************************************************************************
//
// Pointer to EPI SDRAM if available.
//
//*****************************************************************************
uint16_t *g_pui16SDRAM;

//*****************************************************************************
//
// Frame buffer and palette size.
//
//*****************************************************************************
#ifdef OPTREX_DISPLAY
const tRasterDisplayInfo *g_psDisplayMode = &g_sOptrex800x480x75Hz;
#define SCREEN_BPP    16
#else
#ifdef LXD_DISPLAY
const tRasterDisplayInfo *g_psDisplayMode = &g_sLXD640x480x60Hz;
#define SCREEN_BPP    16
#else
#ifdef FORMIKE_DISPLAY
const tRasterDisplayInfo *g_psDisplayMode = &g_sFormike800x480x60Hz;
#define SCREEN_BPP    16
#else
#ifdef INNOLUX_DISPLAY
const tRasterDisplayInfo *g_psDisplayMode = &g_sInnoLux800x480x60Hz;
#define SCREEN_BPP    16
#else
#error Target display type is not defined!
#endif
#endif
#endif
#endif

#define SIZE_BUFFER ((RASTER_WIDTH * RASTER_HEIGHT * SCREEN_BPP) / 8)
#define SIZE_PALETTE ((SCREEN_BPP == 8) ? (256 * 2) : (16 * 2))

//*****************************************************************************
//
// Pointers to the frame buffer into SDRAM.
//
//*****************************************************************************
uint32_t *g_pui32DisplayBuffer;
uint16_t *g_pui16Palette;

#if SCREEN_BPP == 8
#define NUM_PAL_ENTRIES 256
#else
#define NUM_PAL_ENTRIES 16
#endif

const uint32_t g_pulSrcPalette[NUM_PAL_ENTRIES] =
{
    ClrBlack,
    ClrWhite,
    ClrRed,
    0x00FF00,
    ClrBlue,
    ClrYellow,
    ClrMagenta,
    ClrCyan,
    ClrOrange,
    ClrDarkBlue,
    ClrGold,
    ClrOrange,
    ClrCrimson,
    ClrDarkTurquoise,
    ClrDarkGray,
    ClrSilver,
};

//*****************************************************************************
//
// A couple of macros used to extract the dimensions from an image.
//
//*****************************************************************************
#define IMAGE_WIDTH(ptr) ((*(uint16_t *)((uint8_t *)(ptr) + 1)))
#define IMAGE_HEIGHT(ptr) ((*(uint16_t *)((uint8_t *)(ptr) + 3)))

//*****************************************************************************
//
// Graphics context used to show text on the QVGA display.
//
//*****************************************************************************
tContext g_sContext;

//*****************************************************************************
//
// Global interrupt status flags.
//
//*****************************************************************************
static volatile uint32_t g_ui32Flags;
static volatile uint32_t g_ui32FrameCounter;
static volatile uint32_t g_ui32UnderflowCount;

//*****************************************************************************
//
// The image of the TI logo.
//
//*****************************************************************************
const unsigned char g_pucLogo[] =
{
    IMAGE_FMT_4BPP_COMP,
    80, 0,
    75, 0,

    15,
    0x00, 0x00, 0x00,
    0x02, 0x02, 0x0f,
    0x06, 0x05, 0x27,
    0x09, 0x07, 0x3b,
    0x0c, 0x09, 0x4c,
    0x0d, 0x0a, 0x56,
    0x10, 0x0c, 0x68,
    0x13, 0x0f, 0x7a,
    0x15, 0x10, 0x89,
    0x17, 0x11, 0x95,
    0x19, 0x14, 0xa5,
    0x1c, 0x16, 0xb6,
    0x1e, 0x18, 0xc7,
    0x22, 0x1a, 0xdc,
    0x22, 0x1b, 0xe3,
    0x24, 0x1c, 0xed,

    0xfc, 0x07, 0x07, 0x07, 0x07, 0x07, 0x03, 0x04, 0xfe, 0x13, 0xee, 0xee,
    0xee, 0xe9, 0xef, 0xd1, 0x07, 0x07, 0xc2, 0x77, 0x29, 0x04, 0xff, 0xff,
    0xff, 0xe9, 0xff, 0x3c, 0xff, 0xd1, 0x07, 0x07, 0x77, 0x29, 0x04, 0xff,
    0x23, 0xff, 0xff, 0xe9, 0xff, 0xff, 0xd1, 0x07, 0x07, 0xc2, 0x77, 0x29,
    0x04, 0xff, 0xff, 0xff, 0xe9, 0xff, 0x2e, 0xff, 0xd1, 0x05, 0x01, 0x07,
    0xbf, 0x32, 0x04, 0x11, 0xff, 0xff, 0xff, 0xe9, 0xff, 0xff, 0xd1, 0x02,
    0x0e, 0x6f, 0xdc, 0xdc, 0xf7, 0x07, 0xbf, 0x32, 0x04, 0x11, 0xff, 0xff,
    0xff, 0xe9, 0xff, 0xff, 0xd1, 0x02, 0x0e, 0xaf, 0xff, 0xff, 0xf5, 0x07,
    0xbf, 0x32, 0x04, 0x11, 0xff, 0xff, 0xff, 0xe9, 0xff, 0xff, 0xd1, 0x02,
    0x0e, 0xdf, 0xff, 0xff, 0xf2, 0x07, 0xbf, 0x32, 0x04, 0x11, 0xff, 0xff,
    0xff, 0xe9, 0xff, 0xff, 0xd1, 0x01, 0x73, 0x01, 0xc5, 0x07, 0xa7, 0x00,
    0x04, 0x29, 0x11, 0xae, 0x04, 0x03, 0x99, 0xa0, 0x07, 0xbf, 0x72, 0x04,
    0xc5, 0x29, 0x11, 0xff, 0xff, 0xd1, 0x01, 0x08, 0x99, 0x76, 0x70, 0x07,
    0xbf, 0x72, 0x04, 0x29, 0x11, 0xff, 0x20, 0xff, 0xd1, 0x01, 0x07, 0xcb,
    0xbb, 0xbc, 0x40, 0xe1, 0x07, 0xbf, 0x72, 0x04, 0xff, 0xff, 0xff, 0xe9,
    0x1e, 0xff, 0xff, 0xd1, 0x07, 0x07, 0x77, 0x29, 0x04, 0x11, 0xff, 0xff,
    0xff, 0xe9, 0xff, 0xff, 0xd1, 0x07, 0xe1, 0x07, 0x77, 0x29, 0x04, 0xff,
    0xff, 0xff, 0xe9, 0x00, 0xff, 0xff, 0xe6, 0x45, 0x63, 0x00, 0x26, 0x55,
    0x2b, 0x55, 0x64, 0xda, 0x55, 0xe9, 0x51, 0x04, 0xd4, 0x03, 0x00, 0x00,
    0x04, 0xff, 0xff, 0xff, 0xe9, 0xd2, 0x03, 0xf7, 0x00, 0x6f, 0xff, 0xff,
    0xf9, 0xda, 0x62, 0x63, 0xf2, 0x04, 0xd4, 0x00, 0x00, 0x04, 0x54, 0x22,
    0x00, 0xf3, 0x00, 0x9f, 0xff, 0xff, 0xf6, 0x00, 0x8f, 0xb1, 0x74, 0xf2,
    0x04, 0xd4, 0x00, 0x00, 0x04, 0x54, 0x8e, 0x22, 0xd1, 0x00, 0xcf, 0x0a,
    0xd9, 0x62, 0xf2, 0xc6, 0x04, 0xd4, 0x00, 0x00, 0x04, 0x54, 0x22, 0xc0,
    0x45, 0x01, 0x01, 0xd1, 0x01, 0xef, 0x74, 0xf2, 0x04, 0x8c, 0xd4, 0x00,
    0x00, 0x04, 0x54, 0x22, 0x90, 0x03, 0xb6, 0x01, 0xb0, 0xda, 0x62, 0xf2,
    0x04, 0xd4, 0x00, 0x32, 0x00, 0x04, 0x4d, 0x11, 0x60, 0x06, 0x01, 0x90,
    0xd8, 0xda, 0x62, 0xf2, 0x04, 0xd4, 0x00, 0x00, 0x04, 0xcb, 0x4d, 0x11,
    0x30, 0x09, 0x01, 0x50, 0xda, 0x62, 0x62, 0xf2, 0x04, 0xd4, 0x00, 0x00,
    0x04, 0x4d, 0xff, 0x0b, 0xff, 0xfe, 0x10, 0x0c, 0x01, 0x30, 0xda, 0x62,
    0x62, 0xf2, 0x04, 0xd4, 0x00, 0x00, 0x04, 0x4d, 0xda, 0x00, 0xaa, 0xb9,
    0x00, 0x1e, 0xff, 0xff, 0xfe, 0x10, 0x0b, 0x1a, 0xba, 0xaa, 0xdf, 0x62,
    0xf2, 0x04, 0xd4, 0x1a, 0x00, 0x00, 0x04, 0x62, 0x41, 0x80, 0x49, 0x2f,
    0x12, 0xff, 0xff, 0xfb, 0x11, 0x01, 0xef, 0x62, 0xf3, 0xc6, 0x04, 0xd4,
    0x00, 0x00, 0x04, 0x62, 0x41, 0x60, 0x85, 0x49, 0x6f, 0xff, 0xff, 0xf8,
    0x6f, 0xf4, 0x04, 0xc4, 0xd4, 0x5e, 0xff, 0xff, 0x30, 0x49, 0x9f, 0xff,
    0x2b, 0xff, 0xf6, 0x11, 0x06, 0x6b, 0xf7, 0x04, 0xd4, 0x11, 0x00, 0x00,
    0x04, 0x5b, 0xff, 0xfe, 0x10, 0x49, 0x0a, 0xbf, 0xff, 0xff, 0xf3, 0x11,
    0x09, 0x6b, 0xfc, 0x86, 0x02, 0x05, 0x55, 0x55, 0x55, 0xe9, 0xd1, 0x38,
    0x90, 0x5b, 0xff, 0xfb, 0x01, 0x01, 0xef, 0xff, 0xff, 0x55, 0xd1, 0x11,
    0x0c, 0x6c, 0x40, 0x69, 0x0d, 0x14, 0xd5, 0x7c, 0x4a, 0xf9, 0x01, 0x03,
    0x01, 0xb0, 0x19, 0x56, 0x1f, 0x1c, 0xc0, 0x69, 0x09, 0x06, 0xc6, 0xf6,
    0xaa, 0x01, 0x06, 0x01, 0x80, 0x19, 0x3f, 0x1c, 0xfa, 0xb0, 0x69, 0x05,
    0x06, 0xc6, 0xf4, 0x02, 0x22, 0x00, 0x41, 0x08, 0x01, 0x50, 0x01, 0x22,
    0x20, 0x7f, 0x1d, 0x06, 0xb4, 0x00, 0x00, 0x00, 0xcf, 0x05, 0x6d, 0xff,
    0x04, 0xfe, 0xff, 0xfe, 0x10, 0x0b, 0x01, 0x20, 0x0c, 0x81, 0x3f, 0xff,
    0xf9, 0x00, 0x00, 0x00, 0x5f, 0x8f, 0x80, 0x46, 0xfc, 0x00, 0x1f, 0xff,
    0xff, 0xfe, 0x10, 0x40, 0x1e, 0x3f, 0xff, 0xf9, 0x00, 0x00, 0x00, 0x08,
    0xc0, 0x8f, 0x46, 0xfa, 0x00, 0x3f, 0xff, 0xff, 0xfb, 0xc0, 0xda, 0x2e,
    0xf9, 0x00, 0x00, 0x00, 0x00, 0x9f, 0xc0, 0x87, 0x3d, 0xf6, 0x00, 0x5f,
    0xff, 0xff, 0xf8, 0x20, 0x00, 0x6f, 0x47, 0xff, 0xf9, 0x00, 0x00, 0x00,
    0x30, 0x00, 0x08, 0x87, 0x3d, 0xf3, 0x00, 0x9f, 0xff, 0x30, 0xff, 0xf6,
    0xda, 0x36, 0xf9, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x6f, 0x7f, 0x34,
    0xf1, 0x00, 0xcf, 0x18, 0xff, 0xff, 0xf3, 0xda, 0x3e, 0xf9, 0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x05, 0x7f, 0x34, 0xc0, 0x01, 0x0c, 0xef, 0xff,
    0xff, 0xd0, 0xda, 0x3e, 0xf9, 0x00, 0x2c, 0x00, 0x00, 0xe9, 0x6f, 0x77,
    0x2b, 0x90, 0x02, 0x90, 0x01, 0xb0, 0x03, 0x5f, 0xff, 0xff, 0xf9, 0x00,
    0x2c, 0x00, 0x00, 0xe9, 0x09, 0x6f, 0x23, 0x60, 0x06, 0xb0, 0x01, 0x80,
    0xda, 0x46, 0xf9, 0x00, 0x00, 0x00, 0x99, 0xe9, 0x01, 0xdf, 0x67, 0x1a,
    0x30, 0x09, 0x01, 0x61, 0x50, 0xda, 0x4e, 0xf9, 0x00, 0x00, 0x00, 0xe9,
    0x31, 0x00, 0x7f, 0x67, 0x19, 0xfe, 0x10, 0x0b, 0x01, 0x61, 0x20, 0xda,
    0x4e, 0xf9, 0x00, 0x00, 0x00, 0xe9, 0x30, 0x00, 0x2f, 0x67, 0x19, 0xfc,
    0x00, 0x1e, 0xff, 0x08, 0xff, 0xfd, 0x10, 0x0c, 0x67, 0xff, 0xff, 0xf9,
    0x13, 0x00, 0x00, 0x00, 0xe9, 0x00, 0x0a, 0x67, 0x5b, 0x00, 0x3f, 0xee,
    0xee, 0xfb, 0x00, 0x05, 0xee, 0xed, 0x84, 0x6f, 0xf9, 0x00, 0x00, 0x00,
    0xe9, 0x00, 0x05, 0xc0, 0x77, 0x29, 0xf6, 0x00, 0x11, 0x11, 0x11, 0x11,
    0x08, 0x00, 0x00, 0x01, 0x04, 0x57, 0xf9, 0x00, 0x00, 0x44, 0x00, 0xe9,
    0x00, 0x00, 0xcf, 0x6f, 0xff, 0xff, 0x52, 0xf2, 0x5e, 0x06, 0x5e, 0xc8,
    0x42, 0x6e, 0x6f, 0x85, 0x6a, 0xfc, 0x42, 0x25, 0xdf, 0x29, 0xe1, 0x5e,
    0x42, 0x0a, 0x5a, 0xff, 0xff, 0xfd, 0x72, 0x7e, 0x00, 0x21, 0x00, 0x0c,
    0x6a, 0x90, 0x00, 0x00, 0x1c, 0x29, 0x63, 0xc0, 0x5f, 0x5a, 0xff, 0xfc,
    0x40, 0x86, 0x41, 0x50, 0x02, 0x01, 0xf8, 0x01, 0x01, 0xef, 0xff, 0xff,
    0x51, 0xc0, 0x46, 0x1e, 0x59, 0xff, 0xfb, 0x20, 0x8e, 0x84, 0x4b, 0x3f,
    0xff, 0xff, 0x90, 0x02, 0x5f, 0xff, 0x20, 0xff, 0xe1, 0x3e, 0x4f, 0xff,
    0xff, 0xff, 0xfd, 0x62, 0x30, 0x96, 0x54, 0x01, 0x6b, 0xb5, 0x0b, 0x0b,
    0x10, 0xff, 0xff, 0xf6, 0x36, 0x7f, 0xff, 0xff, 0xff, 0x74, 0x70, 0x9e,
    0x5e, 0x1d, 0x03, 0x27, 0x00, 0x00, 0x0e, 0x9f, 0xff, 0xff, 0xd2, 0x07,
    0xbf, 0x1d, 0xbf, 0x00, 0xff, 0xff, 0xfd, 0x96, 0x54, 0x33, 0x33, 0x33,
    0x03, 0x33, 0x32, 0xcf, 0xff, 0xfc, 0x10, 0x07, 0xbf, 0x86, 0x75, 0x4f,
    0xff, 0xff, 0xff, 0xe9, 0xd4, 0xb0, 0xee, 0x07, 0x07, 0x76, 0x0b, 0x02,
    0xe2, 0xc1, 0xfc, 0xee, 0x07, 0x07, 0x77, 0x05, 0x01, 0xe9, 0xd3, 0xe1,
    0xe0, 0x07, 0x07, 0x77, 0x00, 0xcf, 0xff, 0xff, 0xff, 0xdc, 0xe9, 0xd2,
    0x40, 0x07, 0x07, 0x77, 0x00, 0x6f, 0x1b, 0xff, 0xff, 0xff, 0xe9, 0xd1,
    0xf9, 0x07, 0x07, 0x81, 0x77, 0x00, 0x00, 0x1d, 0xff, 0xff, 0xff, 0xe9,
    0xb8, 0xd1, 0xf2, 0x07, 0x07, 0x77, 0x00, 0x00, 0x06, 0x1b, 0xff, 0xff,
    0xff, 0xe9, 0xd1, 0x90, 0x07, 0x07, 0x80, 0x77, 0x00, 0x00, 0x01, 0xdf,
    0xff, 0xff, 0xff, 0x8f, 0xe9, 0xff, 0xff, 0x40, 0x07, 0x07, 0x77, 0x29,
    0x08, 0x6f, 0xff, 0xff, 0xff, 0xe9, 0xff, 0xfe, 0x10, 0xf0, 0x07, 0x07,
    0x77, 0x29, 0x0c, 0xff, 0xff, 0xff, 0x9e, 0xe9, 0xff, 0xfb, 0x07, 0x07,
    0x77, 0x2a, 0x03, 0x13, 0xff, 0xff, 0xff, 0xe9, 0xff, 0xfb, 0x07, 0x07,
    0xc2, 0x77, 0x2b, 0x8f, 0xff, 0xff, 0xff, 0xe9, 0xfb, 0xf0, 0x07, 0x07,
    0x77, 0x2b, 0x0a, 0xff, 0xff, 0xff, 0xbc, 0xe9, 0xfc, 0x07, 0x07, 0x77,
    0x2c, 0xcf, 0xff, 0x03, 0xff, 0xff, 0xff, 0xff, 0xfd, 0x10, 0x07, 0x07,
    0xc2, 0x77, 0x2b, 0x0c, 0xff, 0xff, 0xff, 0xe9, 0x30, 0xf0, 0x07, 0x07,
    0x77, 0x2c, 0xaf, 0xff, 0xff, 0xff, 0x1e, 0xff, 0xff, 0x60, 0x07, 0x07,
    0x77, 0x2c, 0x05, 0x03, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc0, 0x07, 0x07,
    0xc0, 0x77, 0x2d, 0x18, 0xff, 0xff, 0xff, 0xff, 0xf2, 0xf0, 0x07, 0x07,
    0x77, 0x2e, 0x14, 0x79, 0xa9, 0x86, 0x7f, 0x20, 0x07, 0x07, 0x77, 0x2f,
    0x07, 0x07, 0x07, 0xf0, 0x07, 0x07, 0x07, 0x07,
};

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
void
__error__(char *pcFilename, unsigned long ulLine)
{
    //
    // A runtime error was detected so stop here to allow debug.
    //
    while(1)
    {
        //
        // Hang.
        //
    }
}

//*****************************************************************************
//
// The interrupt handler for the LCD controller.  This function merely
// flags error or status interrupts as they are received.
//
//*****************************************************************************
void
LCDIntHandler(void)
{
    uint32_t ui32Status;

    //
    // Get the current interrupt status and clear any active interrupts.
    //
    ui32Status = LCDIntStatus(LCD0_BASE, true);
    LCDIntClear(LCD0_BASE, ui32Status);

    //
    // Increment the frame counter if necessary.
    //
    if(ui32Status & LCD_INT_EOF0)
    {
        g_ui32FrameCounter++;
    }

    //
    // If we saw an underflow interrupt, restart the raster.
    //
    if(ui32Status & LCD_INT_UNDERFLOW)
    {
        g_ui32UnderflowCount++;
        LCDRasterEnable(LCD0_BASE);
    }

    //
    // Update our global flags with the new interrupt status.
    //
    g_ui32Flags |= ui32Status;
}

//*****************************************************************************
//
// Initialize the LCD controller to drive the raster display.
//
//*****************************************************************************
static void
DisplayInit(uint32_t ui32SysClkHz)
{
    //
    // Enable the GPIO peripherals used to interface to the LCD panel.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_LCD0);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOJ);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPION);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOR);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOS);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOT);

    //
    // Configure all the LCD controller pins for hardware control.
    //
    GPIOPinTypeLCD(GPIO_PORTF_BASE, 0x80);
    GPIOPinTypeLCD(GPIO_PORTJ_BASE, 0x7C);
    GPIOPinTypeLCD(GPIO_PORTN_BASE, 0xC0);
#ifdef INNOLUX_DISPLAY
    GPIOPinTypeLCD(GPIO_PORTR_BASE, 0xF9);
#else
    GPIOPinTypeLCD(GPIO_PORTR_BASE, 0xFF);
#endif
    GPIOPinTypeLCD(GPIO_PORTS_BASE, 0xF9);
    GPIOPinTypeLCD(GPIO_PORTT_BASE, 0x0F);

    //
    // Set the pin muxing to ensure that LCD signals appear on the required
    // pins.  Note that we take advantage of the fact that we know the mux
    // selector is F here to allow us to OR the value without first masking
    // off anything that may have been there before.
    //
    HWREG(GPIO_PORTF_BASE + GPIO_O_PCTL) |= 0xF0000000;
    HWREG(GPIO_PORTJ_BASE + GPIO_O_PCTL) |= 0x0FFFFF00;
    HWREG(GPIO_PORTN_BASE + GPIO_O_PCTL) |= 0xFF000000;
#ifdef INNOLUX_DISPLAY
    HWREG(GPIO_PORTR_BASE + GPIO_O_PCTL) |= 0xFFFFF00F;
#else
    HWREG(GPIO_PORTR_BASE + GPIO_O_PCTL) |= 0xFFFFFFFF;
#endif
    HWREG(GPIO_PORTS_BASE + GPIO_O_PCTL) |= 0xFFFFF00F;
    HWREG(GPIO_PORTT_BASE + GPIO_O_PCTL) |= 0x0000FFFF;

    //
    // Configure the LCD controller for raster operation with a pixel clock
    // as close to the requested pixel clock as possible.
    //
    MAP_LCDModeSet(LCD0_BASE, (LCD_MODE_RASTER | LCD_MODE_AUTO_UFLOW_RESTART),
                   g_psDisplayMode->ui32PixClock, ui32SysClkHz);

    //
    // Set the output format for the raster interface.
    //
    MAP_LCDRasterConfigSet(LCD0_BASE, (RASTER_FMT_ACTIVE_PALETTIZED_16BIT |
                           RASTER_NIBBLE_MODE_ENABLED |
                           RASTER_READ_ORDER_REVERSED), 0);


    MAP_LCDRasterTimingSet(LCD0_BASE, &(g_psDisplayMode->sTiming));

    //
    // Configure DMA-related parameters.
    //
    MAP_LCDDMAConfigSet(LCD0_BASE, LCD_DMA_BURST_16 | LCD_DMA_FIFORDY_64_WORDS);

    //
    // If the chosen display has an initialization function, call it now.
    //
    if(g_psDisplayMode->pfnInitDisplay)
    {
        g_psDisplayMode->pfnInitDisplay(ui32SysClkHz);
    }

    //
    // Set up the frame buffer.  Note that we allow this buffer to extend
    // outside the available SRAM.  This allows us to easily test modes where
    // we can't fit the whole frame in memory, realizing, of course, that
    // part of the display will contain crud.
    //
    MAP_LCDRasterFrameBufferSet(LCD0_BASE, 0, g_pui32DisplayBuffer,
                                SIZE_PALETTE + SIZE_BUFFER);

    //
    // Write the palette to the frame buffer.
    //
    MAP_LCDRasterPaletteSet(LCD0_BASE,
#if SCREEN_BPP == 1
                            LCD_PALETTE_SRC_24BIT |LCD_PALETTE_TYPE_1BPP,
#else
#if SCREEN_BPP == 4
                            LCD_PALETTE_SRC_24BIT | LCD_PALETTE_TYPE_4BPP,
#else
#if SCREEN_BPP == 8
                            LCD_PALETTE_SRC_24BIT | LCD_PALETTE_TYPE_8BPP,
#else
#if SCREEN_BPP == 16
                            LCD_PALETTE_SRC_24BIT| LCD_PALETTE_TYPE_DIRECT,
#endif
#endif
#endif
#endif
                            (uint32_t *)g_pui16Palette, g_pulSrcPalette, 0,
                            (SIZE_PALETTE / 2));

    //
    // Enable the LCD interrupts.
    //
    MAP_LCDIntEnable(LCD0_BASE, (LCD_INT_DMA_DONE | LCD_INT_SYNC_LOST |
                     LCD_INT_UNDERFLOW | LCD_INT_EOF0));

    MAP_IntEnable(INT_LCD0);

    //
    // Enable the raster output.
    //
    MAP_LCDRasterEnable(LCD0_BASE);
}

//*****************************************************************************
//
// This function fills the whole display with a given color.
//
//*****************************************************************************
void
FillScreen(uint32_t ui32Color)
{
    tRectangle sRect;

    //
    // Fill the frame buffer with the desired color.
    //
    sRect.i16XMin = 0;
    sRect.i16XMax = GrContextDpyWidthGet(&g_sContext) - 1;
    sRect.i16YMin = 0;
    sRect.i16YMax = GrContextDpyHeightGet(&g_sContext) - 1;
    GrContextForegroundSet(&g_sContext, ui32Color);
    GrRectFill(&g_sContext, &sRect);
}

//*****************************************************************************
//
// This function draws a pattern of diagonal lines on the display.
//
//*****************************************************************************
void
DrawLinePattern(uint32_t ui32Color)
{
    int32_t i32Loop, i32XInc, i32YInc;
    tRectangle sRect;

    //
    // Set the foreground color.
    //
    GrContextForegroundSet(&g_sContext, ui32Color);


    //
    // Draw a rectangle around the whole display.
    //
    sRect.i16XMin = 0;
    sRect.i16YMin = 0;
    sRect.i16XMax = GrContextDpyWidthGet(&g_sContext) - 1;
    sRect.i16YMax = GrContextDpyHeightGet(&g_sContext) - 1;
    GrRectDraw(&g_sContext, &sRect);

    //
    // Determine the spacing of the diagonal lines.
    //
    i32XInc = GrContextDpyWidthGet(&g_sContext) / 20;
    i32YInc = GrContextDpyHeightGet(&g_sContext) / 20;

    //
    // Draw a pattern of lines.
    //
    for(i32Loop = 0; i32Loop < 20; i32Loop++)
    {
        GrLineDraw(&g_sContext, 0, i32YInc * i32Loop, i32XInc * i32Loop,
               (GrContextDpyHeightGet(&g_sContext) - 1));
        GrLineDraw(&g_sContext,
               (GrContextDpyWidthGet(&g_sContext) - 1),
               (GrContextDpyHeightGet(&g_sContext) -
                       (i32YInc * i32Loop + 1)),
               (GrContextDpyWidthGet(&g_sContext) -
                       (i32XInc * i32Loop + 1)),
                0);
    }
}

//*****************************************************************************
//
// A simple example using a raster-mode LCD panel to display graphics.
//
//*****************************************************************************
int
main(void)
{
    uint32_t ui32Val;
    uint32_t ui32SysClk;
    uint32_t ui32X, ui32Y;

    //
    // Set the PLL and system clock to the frequencies needed to allow
    // generation of the required pixel clock.
    //
    ui32SysClk = SysCtlClockFreqSet((SYSCTL_OSC_INT | SYSCTL_USE_PLL |
                                     g_psDisplayMode->ui32VCOFrequency),
                                    g_psDisplayMode->ui32SysClockFrequency);

    //
    // Enable GPIOA for the UART.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    //
    // Set GPIO A0 and A1 as UART pins.
    //
    ui32Val = HWREG(GPIO_PORTA_BASE + GPIO_O_PCTL);
    ui32Val &= 0xFFFFFF00;
    ui32Val |= 0x00000011;
    HWREG(GPIO_PORTA_BASE + GPIO_O_PCTL) = ui32Val;

    //
    // Set GPIO A0 and A1 as UART.
    //
    MAP_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // Initialize the UART as a console for text I/O.
    //
    UARTStdioConfig(0, 115200, ui32SysClk);

    //
    // Enable interrupts.
    //
    MAP_IntMasterEnable();

    //
    // Print hello message to user.
    //
    UARTprintf("\n\nLCD Raster Mode Example\n\n");
    UARTprintf("Display configured for %dx%d at %dbpp.\n",
               RASTER_WIDTH, RASTER_HEIGHT, SCREEN_BPP);
    UARTprintf("System clock is %dMHz\n", ui32SysClk / 1000000);

    //
    // Initialize the SDRAM.
    //
    g_pui16SDRAM = SDRAMInit(ui32SysClk);

    //
    // Was the SDRAM found?
    //
    if(!g_pui16SDRAM)
    {
        UARTprintf("Application requires SDRAM but this is not present!\n");
        while(1);
    }

    //
    // Set the frame buffer pointers as required.
    //
    g_pui32DisplayBuffer = (uint32_t *)g_pui16SDRAM;
    g_pui16Palette = (uint16_t *)g_pui16SDRAM;

    //
    // Initialize the display.
    //
    DisplayInit(ui32SysClk);

    //
    // Initialize the display driver and graphics context.
    //
#if SCREEN_BPP == 1
    GrRaster1BppDriverInit(g_pui32DisplayBuffer);
    GrContextInit(&g_sContext, &g_sGrRaster1BppDriver);
#endif
#if SCREEN_BPP == 4
    GrRaster4BppDriverInit(g_pui32DisplayBuffer);
    GrContextInit(&g_sContext, &g_sGrRaster4BppDriver);
#endif
#if SCREEN_BPP == 8
    GrRaster8BppDriverInit(g_pui32DisplayBuffer);
    GrContextInit(&g_sContext, &g_sGrRaster8BppDriver);
#endif
#if SCREEN_BPP == 16
    GrRaster16BppDriverInit(g_pui32DisplayBuffer);
    GrContextInit(&g_sContext, &g_sGrRaster16BppDriver);
#endif

    //
    // Fill the frame buffer with black.
    //
    FillScreen(ClrBlack);

    //
    // Draw a pattern of red lines on the display.
    //
    DrawLinePattern(ClrWhite);

    //
    // Draw a yellow circle around the center of the display.
    //
    GrContextForegroundSet(&g_sContext, ClrYellow);
    GrCircleDraw(&g_sContext, GrContextDpyWidthGet(&g_sContext) / 2,
                    GrContextDpyHeightGet(&g_sContext) / 2,
                    GrContextDpyWidthGet(&g_sContext) / 5);

    //
    // Determine where to put the logo image so that it is slightly below the
    // center of the display.
    //
    ui32X = (GrContextDpyWidthGet(&g_sContext) - IMAGE_WIDTH(g_pucLogo)) / 2;
    ui32Y = (GrContextDpyHeightGet(&g_sContext) - IMAGE_HEIGHT(g_pucLogo)) / 2;
    ui32Y+= 40;

    //
    // Make sure we draw the logo in red on a white background (this will
    // matter if we're using the 1bpp driver. Otherwise the colors from
    // the image palette will be used).
    //
    GrContextForegroundSet(&g_sContext, ClrRed);
    GrContextBackgroundSet(&g_sContext, ClrBlack);

    //
    // Draw the TI logo in the center of the screen.
    //
    GrImageDraw(&g_sContext, g_pucLogo, ui32X, ui32Y);

    //
    // Display some text.
    //
    GrContextFontSet(&g_sContext, g_psFontCmss28);
    GrContextForegroundSet(&g_sContext, ClrWhite);
    GrStringDrawCentered(&g_sContext, "TivaWare Graphics", -1,
                         GrContextDpyWidthGet(&g_sContext) / 2,
                         (GrContextDpyHeightGet(&g_sContext) / 2) - 30, false);

    //
    // Flush any cached drawing operations.  This is not required by the
    // raster mode driver but remembering to include the call makes the code
    // more portable in case anyone replaces the graphics driver with one
    // which does buffer graphics commands in future.
    //
    GrFlush(&g_sContext);

    //
    // Loop forever.
    //
    while(1)
    {
        //
        // We just sit here and do nothing.
        //
        SysCtlSleep();
    }
}
