//*****************************************************************************
//
// fonttest.c - Simple font testcase.
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
#include "inc/hw_types.h"
#include "inc/hw_memmap.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/fpu.h"
#include "driverlib/pin_map.h"
#include "driverlib/uart.h"
#include "grlib/grlib.h"
#include "grlib/widget.h"
#include "grlib/canvas.h"
#include "grlib/pushbutton.h"
#include "drivers/Kentec320x240x16_ssd2119_spi.h"
#include "drivers/touch.h"
#include "utils/ustdlib.h"
#include "utils/uartstdio.h"
#include "third_party/fonts/ofl/ofl_fonts.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Font Viewer (fontview)</h1>
//!
//! This example displays the contents of a TivaWare Graphics Library font
//! on the board's LCD touchscreen.  By default, the application shows a test
//! font containing ASCII, the Japanese Hiragana and Katakana alphabets,
//! and a group of Korean Hangul characters.
//!
//! The LCD BoosterPack (BOOSTXL-K350QVG-S1) should be installed on
//! BoosterPack 2 interface headers.
//
//*****************************************************************************
//*****************************************************************************
//
// Gloal variable used to store the frequency of the system clock.
//
//*****************************************************************************
uint32_t g_ui32SysClock;

//*****************************************************************************
//
// Change the following to set the font whose characters you want to view if
// no "font.bin" is found in the root directory of the SDCard.
//
//*****************************************************************************
#define FONT_TO_USE g_pFontCjktest20pt

//*****************************************************************************
//
// A pointer to the font we intend to use.  This will be set depending upon
// whether we are using a font from the SDCard or the internal font defined
// above.
//
//*****************************************************************************
const tFont *g_psFont;

//*****************************************************************************
//
// Forward reference to various widget structures.
//
//*****************************************************************************
extern tCanvasWidget g_sHeading;
extern tCanvasWidget g_sBackground;
extern tCanvasWidget g_sBlockNumCanvas;
extern tCanvasWidget g_sCharNumCanvas;
extern tCanvasWidget g_sCharCanvas;
extern tPushButtonWidget g_sBlockIncBtn;
extern tPushButtonWidget g_sBlockDecBtn;
extern tPushButtonWidget g_sCharIncBtn;
extern tPushButtonWidget g_sCharDecBtn;

//*****************************************************************************
//
// Buffers for various strings.
//
//*****************************************************************************
char g_pcBlocks[20];
char g_pcStartChar[32];

//*****************************************************************************
//
// Forward reference to various handlers and internal functions.
//
//*****************************************************************************
void SetBlockNum(uint32_t ui32BlockNum);
void OnBlockButtonPress(tWidget *psWidget);
void OnCharButtonPress(tWidget *psWidget);
void PaintFontGlyphs(tWidget *psWidget, tContext *psContext);

//*****************************************************************************
//
// The canvas widget acting as the background to the display.
//
//*****************************************************************************
Canvas(g_sBackground, WIDGET_ROOT, 0, &g_sHeading,
       &g_sKentec320x240x16_SSD2119, 0, 23, 320, (240 - 23),
       CANVAS_STYLE_FILL, ClrBlack, 0, 0, 0, 0, 0, 0);

//*****************************************************************************
//
// The heading containing the application title.
//
//*****************************************************************************
Canvas(g_sHeading, &g_sBackground, &g_sCharCanvas, &g_sBlockNumCanvas,
       &g_sKentec320x240x16_SSD2119, 0, 0, 320, 23,
       (CANVAS_STYLE_FILL | CANVAS_STYLE_OUTLINE| CANVAS_STYLE_TEXT),
       ClrDarkBlue, ClrWhite, ClrWhite, g_psFontCm20, "fontview", 0, 0);

//*****************************************************************************
//
// The canvas containing the rendered characters.
//
//*****************************************************************************
Canvas(g_sCharCanvas, &g_sBackground, 0, 0,
       &g_sKentec320x240x16_SSD2119, 0, 48, 320, 192,
       CANVAS_STYLE_APP_DRAWN,
       ClrDarkBlue, ClrWhite, ClrWhite, 0, 0, 0, PaintFontGlyphs);

//*****************************************************************************
//
// The canvas widget displaying the maximum and current block number.
//
//*****************************************************************************
Canvas(g_sBlockNumCanvas, &g_sHeading, &g_sCharNumCanvas, 0,
       &g_sKentec320x240x16_SSD2119, 0, 24, 200, 10,
       (CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT),
       ClrBlack, 0, ClrWhite,
       g_psFontFixed6x8, g_pcBlocks, 0, 0);

//*****************************************************************************
//
// The canvas widget displaying the start codepoint for the block.
//
//*****************************************************************************
Canvas(g_sCharNumCanvas, &g_sHeading, &g_sBlockDecBtn, 0,
       &g_sKentec320x240x16_SSD2119, 0, 34, 200, 10,
       (CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT),
       ClrBlack, 0, ClrWhite,
       g_psFontFixed6x8, g_pcStartChar, 0, 0);

//*****************************************************************************
//
// The button used to decrement the block number.
//
//*****************************************************************************
RectangularButton(g_sBlockDecBtn, &g_sHeading, &g_sBlockIncBtn, 0,
                  &g_sKentec320x240x16_SSD2119, 200, 26, 20, 20,
                  (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
                   PB_STYLE_FILL),
                   ClrDarkBlue, ClrRed, ClrWhite, ClrWhite,
                   g_psFontFixed6x8, "<", 0, 0, 0, 0,
                   OnBlockButtonPress);

//*****************************************************************************
//
// The button used to increment the block number.
//
//*****************************************************************************
RectangularButton(g_sBlockIncBtn, &g_sHeading, &g_sCharDecBtn, 0,
                  &g_sKentec320x240x16_SSD2119, 230, 26, 20, 20,
                  (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
                   PB_STYLE_FILL),
                   ClrDarkBlue, ClrRed, ClrWhite, ClrWhite,
                   g_psFontFixed6x8, ">", 0, 0, 0, 0,
                   OnBlockButtonPress);

//*****************************************************************************
//
// The button used to decrement the character row number.
//
//*****************************************************************************
RectangularButton(g_sCharDecBtn, &g_sHeading, &g_sCharIncBtn, 0,
                  &g_sKentec320x240x16_SSD2119, 260, 26, 20, 20,
                  (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
                   PB_STYLE_FILL | PB_STYLE_AUTO_REPEAT),
                   ClrDarkBlue, ClrRed, ClrWhite, ClrWhite,
                   g_psFontFixed6x8, "^", 0, 0, 70, 20,
                   OnCharButtonPress);

//*****************************************************************************
//
// The button used to increment the character row number.
//
//*****************************************************************************
RectangularButton(g_sCharIncBtn, &g_sHeading, 0, 0,
                  &g_sKentec320x240x16_SSD2119, 290, 26, 20, 20,
                  (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
                   PB_STYLE_FILL | PB_STYLE_AUTO_REPEAT),
                   ClrDarkBlue, ClrRed, ClrWhite, ClrWhite,
                   g_psFontFixed6x8, "v", 0, 0, 70, 20,
                   OnCharButtonPress);

//*****************************************************************************
//
// Text codepage mapping functions.
//
//*****************************************************************************
tCodePointMap g_psCodepointMappings[] =
{
    {CODEPAGE_ISO8859_1, CODEPAGE_UNICODE, GrMapISO8859_1_Unicode},
    {CODEPAGE_UTF_8,     CODEPAGE_UNICODE, GrMapUTF8_Unicode},
    {CODEPAGE_UNICODE,   CODEPAGE_UNICODE, GrMapUnicode_Unicode}
};

#define NUM_CHAR_MAPPINGS (sizeof(g_psCodepointMappings)/sizeof(tCodePointMap))

//*****************************************************************************
//
// Default text rendering parameters.  The only real difference between the
// grlib defaults and this set is the addition of a mapping function to allow
// 32 bit Unicode source.
//
//*****************************************************************************
tGrLibDefaults g_psGrLibSettingDefaults =
{
    GrDefaultStringRenderer,
    g_psCodepointMappings,
    CODEPAGE_UTF_8,
    NUM_CHAR_MAPPINGS,
    0
};

//*****************************************************************************
//
// Top left corner of the grid we use to draw the characters.
//
//*****************************************************************************
#define TOP                     50
#define LEFT                    40

//*****************************************************************************
//
// The character cell size used when redrawing the display.
//
//*****************************************************************************
uint32_t g_ui32CellWidth;
uint32_t g_ui32CellHeight;
uint32_t g_ui32LinesPerPage;
uint32_t g_ui32CharsPerLine;
uint32_t g_ui32StartLine;
uint32_t g_ui32NumBlocks;
uint32_t g_ui32StartChar;
uint32_t g_ui32NumBlockChars;
uint32_t g_ui32BlockNum;

//*****************************************************************************
//
// Macros used to derive the screen position given the character's X and Y
// coordinates in the grid.
//
//*****************************************************************************
#define POSX(x) (LEFT + (g_ui32CellWidth / 2) + (g_ui32CellWidth * (x)))
#define POSY(y) (TOP + (g_ui32CellHeight / 2) + (g_ui32CellHeight * (y)))

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
    UARTprintf("Runtime error at line %d of %s!\n", ui32Line, pcFilename);
    while(1)
    {
    }
}
#endif

//*****************************************************************************
//
// This function is called by the graphics library widget manager in the
// context of WidgetMessageQueueProcess whenever the user releases the
// ">" or "<" button.
//
//*****************************************************************************
void
OnBlockButtonPress(tWidget *psWidget)
{
    bool bRedraw;

    //
    // Assume no redraw until we determine otherwise.
    //
    bRedraw = false;

    //
    // Are we incrementing or decrementing the block number?
    //
    if(psWidget == (tWidget *)&g_sBlockIncBtn)
    {
        //
        // We are incrementing.  Have we already reached the top block?
        //
        if((g_ui32BlockNum + 1) < g_ui32NumBlocks)
        {
            //
            // No - increment the block number.
            //
            g_ui32BlockNum++;
            bRedraw = true;
        }
    }
    else
    {
        //
        // We are decrementing.  Are we already showing the first block?
        //
        if(g_ui32BlockNum)
        {
            //
            // No - move back 1 block.
            //
            g_ui32BlockNum--;
            bRedraw = true;
        }
    }

    //
    // If we made a change, set things up to display the new block.
    //
    if(bRedraw)
    {
        SetBlockNum(g_ui32BlockNum);
    }
}

//*****************************************************************************
//
// This function is called by the graphics library widget manager in the
// context of WidgetMessageQueueProcess whenever the user releases the
// "^" or "v" button.
//
//*****************************************************************************
void
OnCharButtonPress(tWidget *psWidget)
{
    bool bRedraw;

    //
    // Assume no redraw until we determine otherwise.
    //
    bRedraw = false;

    //
    // Were we asked to scroll up or down?
    //
    if(psWidget == (tWidget *)&g_sCharIncBtn)
    {
        //
        // Scroll down if there are more characters to display.
        //
        if(((g_ui32StartLine + g_ui32LinesPerPage) * g_ui32CharsPerLine) <
            g_ui32NumBlockChars)
        {
            g_ui32StartLine++;
            bRedraw = true;
        }
    }
    else
    {
        //
        // Scroll up if we're not already showing the first line.
        //
        if(g_ui32StartLine)
        {
            g_ui32StartLine--;
            bRedraw = true;
        }
    }

    //
    // If we made a change, redraw the character area.
    //
    if(bRedraw)
    {
        WidgetPaint((tWidget *)&g_sCharCanvas);
    }
}

//*****************************************************************************
//
// Update the display for a new font block.
//
//*****************************************************************************
void
SetBlockNum(uint32_t ui32BlockNum)
{
    uint32_t ui32Start, ui32Chars;

    //
    // Set the new block number.
    //
    ui32Chars = GrFontBlockCodepointsGet(g_psFont, ui32BlockNum, &ui32Start);

    //
    // If this block exists, update our state.
    //
    if(ui32Chars)
    {
        //
        // Remember details of the new block.
        //
        g_ui32BlockNum = ui32BlockNum;
        g_ui32StartChar = ui32Start;
        g_ui32NumBlockChars = ui32Chars;
        g_ui32StartLine = 0;

        //
        // Update the valid block display and start character.
        //
        usnprintf(g_pcBlocks, sizeof(g_pcBlocks), "Block %d of %d  ",
                  g_ui32BlockNum + 1, g_ui32NumBlocks);
        usnprintf(g_pcStartChar, sizeof(g_pcStartChar), "%d chars from 0x%08x",
                 g_ui32NumBlockChars, g_ui32StartChar);
    }

    //
    // Repaint the display
    //
    WidgetPaint((tWidget *)WIDGET_ROOT);
}

//*****************************************************************************
//
// Configure the UART and its pins.  This must be called before UARTprintf().
//
//*****************************************************************************
void
ConfigureUART(void)
{
    //
    // Enable the GPIO Peripheral used by the UART.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    //
    // Enable UART0
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Configure GPIO Pins for UART mode.
    //
    ROM_GPIOPinConfigure(GPIO_PA0_U0RX);
    ROM_GPIOPinConfigure(GPIO_PA1_U0TX);
    ROM_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, g_ui32SysClock);
}

//*****************************************************************************
//
// Main entry function for the fontview application.
//
//*****************************************************************************
int
main(void)
{
    uint32_t ui32Height, ui32Width;

    //
    // The FPU should be enabled because some compilers will use floating-
    // point registers, even for non-floating-point code.  If the FPU is not
    // enabled this will cause a fault.  This also ensures that floating-
    // point operations could be added to this application and would work
    // correctly and use the hardware floating-point unit.  Finally, lazy
    // stacking is enabled for interrupt handlers.  This allows floating-
    // point instructions to be used within interrupt handlers, but at the
    // expense of extra stack usage.
    //
    FPUEnable();
    FPULazyStackingEnable();

    //
    // Run from the PLL at 120 MHz.
    //
    g_ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                             SYSCTL_OSC_MAIN | SYSCTL_USE_PLL |
                                             SYSCTL_CFG_VCO_480), 120000000);

    //
    // Enable Interrupts
    //
    ROM_IntMasterEnable();

    //
    // Initialize the UART as a console for text I/O.
    //
    ConfigureUART();

    //
    // Print Banner.
    //
    UARTprintf("FontView example running...\n");

    //
    // Initialize the display driver.
    //
    Kentec320x240x16_SSD2119Init(g_ui32SysClock);

    //
    // Initialize the touch screen driver.
    //
    TouchScreenInit(g_ui32SysClock);

    //
    // Set the touch screen event handler.
    //
    TouchScreenCallbackSet(WidgetPointerMessage);

    //
    // Set graphics library text rendering defaults.
    //
    GrLibInit(&g_psGrLibSettingDefaults);

    //
    // Add the compile-time defined widgets to the widget tree.
    //
    WidgetAdd(WIDGET_ROOT, (tWidget *)&g_sBackground);

    //
    // Paint the widget tree to make sure they all appear on the display.
    //
    WidgetPaint(WIDGET_ROOT);

    //
    // See if there is a file called "font.bin" in the root directory of the
    // SDCard.  If there is, use this as the font to display rather than the
    // one defined via FONT_TO_USE.
    //
    g_psFont = FONT_TO_USE;

    //
    // How big is the test font character cell?
    //
    ui32Height = GrFontHeightGet(g_psFont);
    ui32Width = GrFontMaxWidthGet(g_psFont);

    //
    // Determine the size of the character cell to use for this font. We
    // limit the cell size such that we either get 8 or 16 characters per line.
    //
    g_ui32CharsPerLine = (ui32Width > ((320 - LEFT) / 16)) ? 8 : 16;
    g_ui32CellWidth = (320 - LEFT) / g_ui32CharsPerLine;
    g_ui32CellHeight = ui32Height + 4;
    g_ui32LinesPerPage = (240 - TOP) / g_ui32CellHeight;
    g_ui32StartLine = (0x20 / g_ui32CharsPerLine);

    //
    // Get the number of blocks in the font and set up to display the content
    // of the first.
    //
    g_ui32NumBlocks = GrFontNumBlocksGet(g_psFont);
    SetBlockNum(0);

    //
    // Loop forever, processing widget messages.
    //
    while(1)
    {
        //
        // Process any messages from or for the widgets.
        //
        WidgetMessageQueueProcess();
    }
}
//*****************************************************************************
//
// Paints the main font glyph section of the display.
//
//*****************************************************************************
void
PaintFontGlyphs(tWidget *psWidget, tContext *psContext)
{
    uint32_t ui32X, ui32Y, ui32Char;
    tRectangle sRect;
    tCanvasWidget *psCanvas;
    char pcBuffer[6];

    //
    // Tell the graphics library we will be using UTF-8 text for now.
    //
    GrStringCodepageSet(psContext, CODEPAGE_UTF_8);

    //
    // Erase the background.
    //
    psCanvas = (tCanvasWidget *)psWidget;
    GrContextForegroundSet(psContext, psCanvas->ui32FillColor);
    sRect = psCanvas->sBase.sPosition;
    GrRectFill(psContext, &sRect);

    //
    // Draw the character indices.
    //
    GrContextForegroundSet(psContext, ClrYellow);
    GrContextFontSet(psContext, g_psFontFixed6x8);

    for(ui32Y = 0; ui32Y < g_ui32LinesPerPage; ui32Y++ )
    {
        usprintf(pcBuffer, "%06x", g_ui32StartChar +
                 ((ui32Y + g_ui32StartLine) * g_ui32CharsPerLine));
        GrStringDraw(psContext, pcBuffer, -1, 0, POSY(ui32Y), 0);
    }

    //
    // Tell the graphics library to render pure, 32 bit Unicode source
    // text.
    //
    GrStringCodepageSet(psContext, CODEPAGE_UNICODE);

    //
    // Draw the required characters at their positions in the grid.
    //
    GrContextFontSet(psContext, g_psFont);
    GrContextForegroundSet(psContext, ClrWhite);

    for(ui32Y = 0; ui32Y < g_ui32LinesPerPage; ui32Y++)
    {
        for(ui32X = 0; ui32X < g_ui32CharsPerLine; ui32X++)
        {
            //
            // Which character are we about to show?
            //
            ui32Char = g_ui32StartChar +
                     ((g_ui32StartLine + ui32Y) * g_ui32CharsPerLine) + ui32X;

            //
            // Fill the character cell with the background color.
            //
            sRect.i16XMin = LEFT + (ui32X * g_ui32CellWidth);
            sRect.i16YMin = TOP + (ui32Y * g_ui32CellHeight);
            sRect.i16XMax = sRect.i16XMin + g_ui32CellWidth;
            sRect.i16YMax = sRect.i16YMin + g_ui32CellHeight;
            GrContextForegroundSet(psContext, psCanvas->ui32FillColor);
            GrRectFill(psContext, &sRect);
            GrContextForegroundSet(psContext, ClrWhite);

            //
            // Have we run off the end of the end of the block?
            //
            if((ui32Char - g_ui32StartChar) < g_ui32NumBlockChars)
            {
                //
                // No - display the character.
                //

                GrStringDrawCentered(psContext, (char const *)&ui32Char, 4,
                                     POSX(ui32X), POSY(ui32Y), 0);
            }
        }
    }
}
