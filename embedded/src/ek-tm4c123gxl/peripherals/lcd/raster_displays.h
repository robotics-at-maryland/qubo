//*****************************************************************************
//
// raster_displays.h - Public header file for the raster display timing module.
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
// This is part of revision 2.1.3.156 of the Tiva Firmware Development Package.
//
//*****************************************************************************
#ifndef RASTER_DISPLAYS_H_
#define RASTER_DISPLAYS_H_

//*****************************************************************************
//
// A structure containing information that can be used to configure the LCD
// controller to drive a particular display.  This structure contains
// recommended system clock, pixel clock and raster control signal timings
// along with a description of the display to which the structure refers and,
// optionally, a pointer to a function which the application should call to
// initialize the display.
//
//*****************************************************************************
typedef struct
{
    //
    // A text description of the display mode.
    //
    char *pcMode;

    //
    // The required pixel clock for the display mode in Hz.
    //
    uint32_t ui32PixClock;

    //
    // The PLL VCO frequency recommended to allow the desired pixel clock
    // frequency to be achieved using an integer system clock divider.
    //
    uint32_t ui32VCOFrequency;

    //
    // The recommended system clock frequency to set to allow the pixel clock
    // to be achieved using an integer system clock divider.  Other values are
    // possible so this is merely a recommendation.
    //
    uint32_t ui32SysClockFrequency;

    //
    // LCD controller raster display timings for the display mode.
    //
    tLCDRasterTiming sTiming;

    //
    // A pointer to an additional, display-specific function which, of not
    // NULL, must be called to perform display initialization prior to
    // enabling the LCD controller raster engine.
    //
    void (*pfnInitDisplay)(uint32_t);
}
tRasterDisplayInfo;

//*****************************************************************************
//
// Modes exported from the raster_display.c file.
//
//*****************************************************************************
extern const tRasterDisplayInfo g_sOptrex800x480x75Hz;
extern const tRasterDisplayInfo g_sFormike800x480x60Hz;
extern const tRasterDisplayInfo g_sLXD640x480x60Hz;
extern const tRasterDisplayInfo g_sInnoLux800x480x60Hz;

#endif /* RASTER_DISPLAYS_H_ */
