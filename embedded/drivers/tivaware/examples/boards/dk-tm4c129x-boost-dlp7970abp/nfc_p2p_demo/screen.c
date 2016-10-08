//*****************************************************************************
//
// screen.c - Touch Screen movements / Canvases for NFC P2P Demo.
//
// Copyright (c) 2014-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the DK-TM4C129X Firmware Package.
//
//*****************************************************************************
#include <stdint.h>
#include <stdbool.h>
#include "grlib/grlib.h"
#include "grlib/widget.h"
#include "grlib/keyboard.h"
#include "grlib/canvas.h"
#include "grlib/pushbutton.h"
#include "drivers/frame.h"
#include "drivers/kentec320x240x16_ssd2119.h"
#include "drivers/pinout.h"
#include "drivers/touch.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "utils/ustdlib.h"
#include "screen.h"
#include "images.h"

//*****************************************************************************
//
// Defines for setting up the system tick clock.
//
//*****************************************************************************
#define SYSTEM_TICK_MS          10
#define SYSTEM_TICK_S           100

//*****************************************************************************
//
// Pre-declaration of the functions used by the graphical objects.
//
//*****************************************************************************
static void KeyEvent(tWidget *psWidget, uint32_t ui32Key, uint32_t ui32Event);

//*****************************************************************************
//
// The animation delay passed to SysCtlDelay().
//
//*****************************************************************************
#define SCREEN_ANIMATE_DELAY    0x10000

//*****************************************************************************
//
// Minimum change to be a swipe action.
//
//*****************************************************************************
#define SWIPE_MIN_DIFF          40

//*****************************************************************************
//
// Screen saver timeout.
//
//*****************************************************************************
volatile uint32_t g_ui32ScreenSaver;

//*****************************************************************************
//
// Screen on before entering keyboard. Must be set before displaying keyboard
//
//*****************************************************************************
volatile int32_t g_i32PreviousScreenIdx = 0;

//*****************************************************************************
//
// State information for the toggle buttons used in the settings panel.
//
//*****************************************************************************
typedef struct
{
    //
    // The outside area of the button.
    //
    tRectangle sRectContainer;

    //
    // The actual button area.
    //
    tRectangle sRectButton;

    //
    // The text for the on position.
    //
    const char *pcOn;

    //
    // The text for the off position.
    //
    const char *pcOff;

    //
    // The label for the button.
    //
    const char *pcLabel;
}
tButtonToggle;

//
// Global graphic context for the application.
//
tContext g_sContext;

//*****************************************************************************
//
// Defines for the basic screen area used by the application.
//
//*****************************************************************************
#define BG_MIN_X                8
#define BG_MAX_X                (320 - 8)
#define BG_MIN_Y                24
#define BG_MAX_Y                (240 - 8)
#define BG_COLOR_SETTINGS       ClrGray
#define BG_COLOR_MAIN           ClrBlack

//*****************************************************************************
//
// The canvas widget acting as the background to the Summary Screen
//
//*****************************************************************************
extern tCanvasWidget g_sSummaryBackground;

char g_pcPayloadLine8[60]="";
Canvas(g_sPayloadLine8, &g_sSummaryBackground, 0, 0,
       &g_sKentec320x240x16_SSD2119, 20, 210, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine8, 0, 0);

char g_pcPayloadLine7[60]="";
Canvas(g_sPayloadLine7, &g_sSummaryBackground, &g_sPayloadLine8, 0,
       &g_sKentec320x240x16_SSD2119, 20, 195, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine7, 0, 0);

char g_pcPayloadLine6[60]="";
Canvas(g_sPayloadLine6, &g_sSummaryBackground, &g_sPayloadLine7, 0,
       &g_sKentec320x240x16_SSD2119, 20, 180, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine6, 0, 0);

char g_pcPayloadLine5[60]="";
Canvas(g_sPayloadLine5, &g_sSummaryBackground, &g_sPayloadLine6, 0,
       &g_sKentec320x240x16_SSD2119, 20, 165, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine5, 0, 0);

char g_pcPayloadLine4[60]="";
Canvas(g_sPayloadLine4, &g_sSummaryBackground, &g_sPayloadLine5, 0,
       &g_sKentec320x240x16_SSD2119, 20, 150, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine4, 0, 0);

char g_pcPayloadLine3[60]="";
Canvas(g_sPayloadLine3, &g_sSummaryBackground, &g_sPayloadLine4, 0,
       &g_sKentec320x240x16_SSD2119, 20, 135, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine3, 0, 0);

char g_pcPayloadLine2[60]="";
Canvas(g_sPayloadLine2, &g_sSummaryBackground, &g_sPayloadLine3, 0,
       &g_sKentec320x240x16_SSD2119, 20, 120, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine2, 0, 0);

char g_pcPayloadLine1[60]="";
Canvas(g_sPayloadLine1, &g_sSummaryBackground, &g_sPayloadLine2, 0,
       &g_sKentec320x240x16_SSD2119, 20, 105, 285, 10,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontFixed6x8, g_pcPayloadLine1, 0, 0);

char g_pcPayloadTitle[]="Payload:";
Canvas(g_sPayloadTitle, &g_sSummaryBackground, &g_sPayloadLine1, 0,
       &g_sKentec320x240x16_SSD2119, 20, 75, 80, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcPayloadTitle, 0, 0);

char g_pcTagType[60]="";
Canvas(g_sTag, &g_sSummaryBackground, &g_sPayloadTitle, 0,
       &g_sKentec320x240x16_SSD2119, 110, 40, 190, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcTagType, 0, 0);

char g_pcTagTitle[]="TagType:";
Canvas(g_sTagTitle, &g_sSummaryBackground, &g_sTag, 0,
       &g_sKentec320x240x16_SSD2119, 20, 40, 80, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcTagTitle, 0, 0);

Canvas(g_sSummaryBackground, WIDGET_ROOT, 0, &g_sTagTitle,
       &g_sKentec320x240x16_SSD2119, BG_MIN_X, BG_MIN_Y,
       BG_MAX_X - BG_MIN_X,
       BG_MAX_Y - BG_MIN_Y, CANVAS_STYLE_FILL,
       BG_COLOR_MAIN, ClrWhite, ClrWhite, g_psFontCmss20,
       0, 0, 0);


//*****************************************************************************
//
// The canvas widget acting as the background to the Details Screen
//
//*****************************************************************************
extern tCanvasWidget g_sDetailsBackground;

char g_pcHeaderLine5[60]="";
Canvas(g_sHeaderLine5, &g_sDetailsBackground, 0, 0,
       &g_sKentec320x240x16_SSD2119, 20, 200, 285, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcHeaderLine5, 0, 0);

char g_pcHeaderLine4[60]="";
Canvas(g_sHeaderLine4, &g_sDetailsBackground, &g_sHeaderLine5, 0,
       &g_sKentec320x240x16_SSD2119, 20, 170, 285, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcHeaderLine4, 0, 0);

char g_pcHeaderLine3[60]="";
Canvas(g_sHeaderLine3, &g_sDetailsBackground, &g_sHeaderLine4, 0,
       &g_sKentec320x240x16_SSD2119, 20, 140, 285, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcHeaderLine3, 0, 0);

char g_pcHeaderLine2[60]="";
Canvas(g_sHeaderLine2, &g_sDetailsBackground, &g_sHeaderLine3, 0,
       &g_sKentec320x240x16_SSD2119, 20, 110, 285, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcHeaderLine2, 0, 0);

char g_pcHeaderLine1[60]="";
Canvas(g_sHeaderLine1, &g_sDetailsBackground, &g_sHeaderLine2, 0,
       &g_sKentec320x240x16_SSD2119, 20, 80, 285, 30,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcHeaderLine1, 0, 0);

char g_pcHeaderTitle[]="Header Info:";
Canvas(g_sHeaderTitle, &g_sDetailsBackground, &g_sHeaderLine1, 0,
       &g_sKentec320x240x16_SSD2119, 20, 40, 130, 40,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT |
       CANVAS_STYLE_TEXT_OPAQUE, BG_COLOR_MAIN, ClrWhite, ClrWhite,
       g_psFontCmss20, g_pcHeaderTitle, 0, 0);

Canvas(g_sDetailsBackground, WIDGET_ROOT, 0, &g_sHeaderTitle,
       &g_sKentec320x240x16_SSD2119, BG_MIN_X, BG_MIN_Y,
       BG_MAX_X - BG_MIN_X,
       BG_MAX_Y - BG_MIN_Y, CANVAS_STYLE_FILL,
       BG_COLOR_MAIN, ClrWhite, ClrWhite, g_psFontCmss20,
       0, 0, 0);

//*****************************************************************************
//
// Settings panel graphical variables.
//
//*****************************************************************************
extern tCanvasWidget g_sStatusPanel;

void OnTINFCButton(tWidget *psWidget);
RectangularButton(g_sTINFCButton, &g_sStatusPanel, 0, 0,
                  &g_sKentec320x240x16_SSD2119, 185, 180, 110, 50,
                  (PB_STYLE_TEXT | PB_STYLE_FILL | PB_STYLE_OUTLINE |
                   PB_STYLE_TEXT_OPAQUE), TI_RED, TI_PANTONE1807, 0, TI_WHITE,
                  g_psFontCm16, "Visit Website", 0, 0, 0, 0,
                  OnTINFCButton);

void OnEchoNFCButton(tWidget *psWidget);
RectangularButton(g_sEchoNFCButton, &g_sStatusPanel, &g_sTINFCButton, 0,
                  &g_sKentec320x240x16_SSD2119, 20, 180, 110, 50,
                  (PB_STYLE_TEXT | PB_STYLE_FILL | PB_STYLE_OUTLINE |
                   PB_STYLE_TEXT_OPAQUE), TI_RED, TI_PANTONE1807, 0, TI_WHITE,
                  g_psFontCm16, "Echo Tag", 0, 0, 0, 0,
                  OnEchoNFCButton);

char g_pcStatusPanelLine2[]="NFC P2P Demo";
Canvas(g_sStatusLine2, &g_sStatusPanel, &g_sEchoNFCButton, 0,
       &g_sKentec320x240x16_SSD2119, 20, 100, 260, 30,
       CANVAS_STYLE_TEXT, TI_GRAY, ClrWhite, TI_BLACK,
       g_psFontCmss20, g_pcStatusPanelLine2, 0, 0);

char g_pcStatusPanelLine1[]="Tiva C Series";
Canvas(g_sStatusLine1, &g_sStatusPanel, &g_sStatusLine2, 0,
       &g_sKentec320x240x16_SSD2119, 20, 70, 260, 30,
       CANVAS_STYLE_TEXT, TI_GRAY, ClrWhite, TI_BLACK,
       g_psFontCmss20, g_pcStatusPanelLine1, 0, 0);

//
// Background of the settings panel.
//
Canvas(g_sStatusPanel, WIDGET_ROOT, 0, &g_sStatusLine1,
       &g_sKentec320x240x16_SSD2119, BG_MIN_X, BG_MIN_Y,
       BG_MAX_X - BG_MIN_X, BG_MAX_Y - BG_MIN_Y,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT_RIGHT |
       CANVAS_STYLE_TEXT_TOP, TI_RED, TI_GRAY, ClrWhite, g_psFontCmss20,
       0, 0, 0);

//*****************************************************************************
//
// Keyboard
//
//*****************************************************************************

//
// Keyboard cursor blink rate.
//
#define KEYBOARD_BLINK_RATE     100

//
// The current string pointer for the keyboard.
//
static char *g_pcKeyStr;

//
// The current string index for the keyboard.
//
static uint32_t g_ui32StringIdx;

//
// A place holder string used when nothing is being displayed on the keyboard.
//
static const char g_cTempStr = 0;

//
// The current string width for the keyboard in pixels.
//
static int32_t g_i32StringWidth;

//
// The cursor blink counter.
//
static volatile uint32_t g_ui32CursorDelay;

extern tCanvasWidget g_sKeyboardBackground;

//
// The keyboard widget used by the application.
//
Keyboard(g_sKeyboard, &g_sKeyboardBackground, 0, 0,
         &g_sKentec320x240x16_SSD2119, 8, 90, 300, 140,
         KEYBOARD_STYLE_FILL | KEYBOARD_STYLE_AUTO_REPEAT |
         KEYBOARD_STYLE_PRESS_NOTIFY | KEYBOARD_STYLE_RELEASE_NOTIFY |
         KEYBOARD_STYLE_BG,
         ClrBlack, ClrGray, ClrDarkGray, ClrGray, ClrBlack, g_psFontCmss14,
         100, 100, NUM_KEYBOARD_US_ENGLISH, g_psKeyboardUSEnglish, KeyEvent);

//
// The keyboard text entry area.
//
Canvas(g_sKeyboardText, &g_sKeyboardBackground, &g_sKeyboard, 0,
       &g_sKentec320x240x16_SSD2119, BG_MIN_X, BG_MIN_Y,
       BG_MAX_X - BG_MIN_X, 60,
       CANVAS_STYLE_FILL | CANVAS_STYLE_TEXT | CANVAS_STYLE_TEXT_LEFT,
       ClrBlack, ClrWhite, ClrWhite, g_psFontCmss24, &g_cTempStr, 0 ,0 );

//
// The full background for the keyboard when it is takes over the screen.
//
Canvas(g_sKeyboardBackground, WIDGET_ROOT, 0, &g_sKeyboardText,
       &g_sKentec320x240x16_SSD2119, BG_MIN_X, BG_MIN_Y,
       BG_MAX_X - BG_MIN_X, BG_MAX_Y - BG_MIN_Y,
       CANVAS_STYLE_FILL, ClrBlack, ClrWhite, ClrWhite, 0, 0, 0 ,0 );

//*****************************************************************************
//
// The main control paths for changing screens.
//
//*****************************************************************************
#define NUM_SCREENS             4

//
// Screen index values.
//
#define SCREEN_SUMMARY          0
#define SCREEN_TI               1
#define SCREEN_DETAILS          2
#define SCREEN_KEYBOARD         3

static struct
{
    tWidget *psWidget;
    uint32_t ui32Up;
    uint32_t ui32Down;
    uint32_t ui32Left;
    uint32_t ui32Right;
}
g_sScreens[NUM_SCREENS] =
{
    {
        (tWidget *)&g_sSummaryBackground,
        SCREEN_SUMMARY, SCREEN_TI, SCREEN_DETAILS, SCREEN_DETAILS
    },
    {
        (tWidget *)&g_sStatusPanel,
        SCREEN_SUMMARY, SCREEN_TI, SCREEN_TI, SCREEN_TI
    },
    {
        (tWidget *)&g_sDetailsBackground,
        SCREEN_DETAILS, SCREEN_TI, SCREEN_SUMMARY, SCREEN_SUMMARY
    },
    {
        (tWidget *)&g_sKeyboardBackground,
        SCREEN_TI, SCREEN_TI, SCREEN_TI, SCREEN_TI
    }
};

//
// The current active screen index.
//
static uint32_t g_i32ScreenIdx;

//*****************************************************************************
//
// The state of the direction control for the application.
//
//*****************************************************************************
static struct
{
    //
    // The initial touch location.
    //
    int32_t i32InitX;
    int32_t i32InitY;

    //
    // The current movement that was detected.
    //
    enum
    {
        iSwipeUp,
        iSwipeDown,
        iSwipeLeft,
        iSwipeRight,
        iSwipeNone,
    }
    eMovement;

    //
    // Holds if the swipe detection is enabled.
    //
    bool bEnable;
}
g_sSwipe;

//*****************************************************************************
//
// The screen buttons state structure.
//
//*****************************************************************************
struct
{
    //
    // Indicates if an on-screen buttons are enabled.
    //
    bool bEnabled;

    //
    // Indicates if an on-screen buttons can be pressed.
    //
    bool bActive;

    //
    // The X position of an on-screen button.
    //
    int32_t i32X;

    //
    // The Y position of an on-screen button.
    //
    int32_t i32Y;

    //
    // The delay time before the on-screen button is removed.
    //
    volatile uint32_t ui32Delay;
}
g_sButtons;

//*****************************************************************************
//
// Handle keyboard updates.
//
//*****************************************************************************
void
HandleKeyboard(void)
{
    //
    // Nothing to do if the keyboard is not active.
    //
    if(g_i32ScreenIdx != SCREEN_KEYBOARD)
    {
        return;
    }

    //
    // If the mid value is hit then clear the cursor.
    //
    if(g_ui32CursorDelay == KEYBOARD_BLINK_RATE / 2)
    {
        GrContextForegroundSet(&g_sContext, ClrBlack);

        //
        // Keep the counter moving now that the clearing has been handled.
        //
        g_ui32CursorDelay--;
    }
    else if(g_ui32CursorDelay == 0)
    {
        GrContextForegroundSet(&g_sContext, ClrWhite);

        //
        // Reset the blink delay now that the drawing of the cursor has been
        // handled.
        //
        g_ui32CursorDelay = KEYBOARD_BLINK_RATE;
    }
    else
    {
        return;
    }

    //
    // Draw the cursor only if it is time.
    //
    GrLineDrawV(&g_sContext, BG_MIN_X + g_i32StringWidth , BG_MIN_Y + 20,
                BG_MIN_Y + 40);
}

//*****************************************************************************
//
// Draw the pop up buttons on the screen.
//
//*****************************************************************************
static void
DrawButtons(int32_t i32Offset, bool bClear)
{
    static const tRectangle sRectTop =
    {
        140,
        BG_MIN_Y,
        171,
        BG_MIN_Y + 10,
    };
    static const tRectangle sRectRight =
    {
        BG_MAX_X - 11,
        BG_MIN_Y - 20 + ((BG_MAX_Y - BG_MIN_Y) / 2),
        BG_MAX_X,
        BG_MIN_Y - 20 + ((BG_MAX_Y - BG_MIN_Y) / 2) + 40,
    };
    static const tRectangle sRectLeft =
    {
        BG_MIN_X,
        BG_MIN_Y - 20 + ((BG_MAX_Y - BG_MIN_Y) / 2),
        BG_MIN_X + 10,
        BG_MIN_Y - 20 + ((BG_MAX_Y - BG_MIN_Y) / 2) + 40,
    };

    //
    // Only draw if they are enabled.
    //
    if(g_sButtons.bEnabled == false)
    {
        return;
    }

    //
    // Draw the three pop up buttons.
    //
    if(g_i32ScreenIdx == SCREEN_SUMMARY || g_i32ScreenIdx == SCREEN_DETAILS)
    {
        GrContextForegroundSet(&g_sContext, ClrBlack);
        GrContextBackgroundSet(&g_sContext, ClrGray);

        GrRectFill(&g_sContext, &sRectRight);
        GrRectFill(&g_sContext, &sRectLeft);

        if(bClear == false)
        {
            GrLineDrawH(&g_sContext, 140, 171, BG_MIN_Y + 10 + i32Offset);

            GrImageDraw(&g_sContext, g_pui8DownTabImage, 140,
                        BG_MIN_Y + i32Offset);

            GrTransparentImageDraw(&g_sContext, g_pui8RightImage,
                                   BG_MAX_X - 10 + i32Offset,
                                   BG_MIN_Y - 20 + ((BG_MAX_Y - BG_MIN_Y) / 2),
                                   1);
            GrTransparentImageDraw(&g_sContext, g_pui8LeftImage,
                                   BG_MIN_X - i32Offset,
                                   BG_MIN_Y - 20 + ((BG_MAX_Y - BG_MIN_Y) / 2),
                                   1);
        }
        else
        {
            GrRectFill(&g_sContext, &sRectTop);
        }
    }
    else if(g_i32ScreenIdx == SCREEN_TI)
    {
        GrContextForegroundSet(&g_sContext, ClrGray);
        GrContextBackgroundSet(&g_sContext, ClrWhite);
        if(bClear == false)
        {
            GrLineDrawH(&g_sContext, 140, 171, BG_MAX_Y - 11 - i32Offset);
            GrImageDraw(&g_sContext, g_pui8UpTabImage, 140,
                        BG_MAX_Y - 10 - i32Offset);
        }
    }
}

//*****************************************************************************
//
// Disable the pop up buttons.
//
//*****************************************************************************
static void
ButtonsDisable(void)
{
    g_sButtons.bEnabled = false;
    g_sButtons.bActive = false;
}

//*****************************************************************************
//
// Handle the animation when switching between screens.
//
//*****************************************************************************
void
AnimatePanel(uint32_t ui32Color)
{
    int32_t i32Idx;

    GrContextForegroundSet(&g_sContext, ui32Color);

    if(g_i32ScreenIdx == SCREEN_DETAILS)
    {
        for(i32Idx = BG_MAX_Y; i32Idx >= BG_MIN_Y; i32Idx--)
        {
            GrLineDrawH(&g_sContext, BG_MIN_X, BG_MAX_X, i32Idx);


            if(i32Idx == 40)
            {
                WidgetPaint((tWidget *)&g_sHeaderTitle);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 70)
            {
                WidgetPaint((tWidget *)&g_sHeaderLine1);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 100)
            {
                WidgetPaint((tWidget *)&g_sHeaderLine2);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 130)
            {
                WidgetPaint((tWidget *)&g_sHeaderLine3);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 160)
            {
                WidgetPaint((tWidget *)&g_sHeaderLine4);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 190)
            {
                WidgetPaint((tWidget *)&g_sHeaderLine5);
                WidgetMessageQueueProcess();
            }

            SysCtlDelay(SCREEN_ANIMATE_DELAY);
        }
    }
    else if(g_i32ScreenIdx == SCREEN_SUMMARY)
    {
        for(i32Idx = BG_MAX_Y; i32Idx >= BG_MIN_Y; i32Idx--)
        {
            GrLineDrawH(&g_sContext, BG_MIN_X, BG_MAX_X, i32Idx);

            if(i32Idx == 210)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine8);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 195)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine7);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 180)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine6);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 165)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine5);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 150)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine4);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 135)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine3);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 120)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine2);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 105)
            {
                WidgetPaint((tWidget *)&g_sPayloadLine1);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 75)
            {
                WidgetPaint((tWidget *)&g_sPayloadTitle);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 40)
            {
                WidgetPaint((tWidget *)&g_sTag);
                WidgetMessageQueueProcess();
                WidgetPaint((tWidget *)&g_sTagTitle);
                WidgetMessageQueueProcess();
            }

            SysCtlDelay(SCREEN_ANIMATE_DELAY);
        }
    }
    else if(g_i32ScreenIdx == SCREEN_TI)
    {
        for(i32Idx = BG_MIN_Y; i32Idx < BG_MAX_Y; i32Idx++)
        {
            GrLineDrawH(&g_sContext, BG_MIN_X, BG_MAX_X, i32Idx);

            if (i32Idx == 100)
            {
                GrImageDraw(&g_sContext, g_pui8TILogo, BG_MIN_X,
                        BG_MIN_Y);
            }
            else if(i32Idx == 140)
            {
                WidgetPaint((tWidget *)&g_sStatusLine1);
                WidgetMessageQueueProcess();
                GrContextForegroundSet(&g_sContext, ui32Color);
            }
            else if(i32Idx == 170)
            {
                //DrawToggle(&sProxyToggle, g_sConfig.bProxyEnabled);
                WidgetPaint((tWidget *)&g_sStatusLine2);
                GrContextForegroundSet(&g_sContext, ui32Color);
                WidgetMessageQueueProcess();
            }
            else if(i32Idx == 230)
            {
                WidgetPaint((tWidget *)&g_sTINFCButton);
                WidgetPaint((tWidget *)&g_sEchoNFCButton);
                GrContextForegroundSet(&g_sContext, ui32Color);
                WidgetMessageQueueProcess();
            }
            SysCtlDelay(SCREEN_ANIMATE_DELAY);
        }
    }
}

//*****************************************************************************
//
// Animate Buttons.
//
//*****************************************************************************
void
AnimateButtons(bool bInit)
{
    if(bInit)
    {
        g_sButtons.i32X = 0;
        g_sButtons.i32Y = 0;
        g_sButtons.bEnabled = true;
        g_sButtons.bActive = false;
        g_sButtons.ui32Delay = 0;
    }
    else if(g_sButtons.bEnabled == false)
    {
        //
        // Just return if the buttons are not on screen.
        //
        return;
    }

    if(g_sButtons.ui32Delay == 0)
    {
        g_sButtons.ui32Delay = 6;

        GrContextForegroundSet(&g_sContext, ClrBlack);
        GrContextBackgroundSet(&g_sContext, ClrGray);

        if((bInit == false) || (g_sButtons.bActive == true))
        {
            //
            // Update the buttons.
            //
            DrawButtons(g_sButtons.i32X - g_sButtons.i32Y, true);

            if(g_sButtons.i32X < 3)
            {
                g_sButtons.i32X++;
            }
            else
            {
                g_sButtons.i32Y++;
            }
        }

        if(g_sButtons.bActive == false)
        {
            //
            // Update the buttons.
            //
            DrawButtons(g_sButtons.i32X - g_sButtons.i32Y, false);

            if(g_sButtons.i32Y >= 3)
            {
                g_sButtons.bActive = true;
                g_sButtons.ui32Delay = 6;
            }
        }
        else if((g_i32ScreenIdx == SCREEN_SUMMARY) ||
                (g_i32ScreenIdx == SCREEN_DETAILS))
        {
            ButtonsDisable();
        }
    }
}

//*****************************************************************************
//
// Clears the full screen.
//
//*****************************************************************************
void
ClearScreen(tContext *psContext)
{
    static const tRectangle sRect =
    {
        0,
        0,
        319,
        239,
    };

    GrRectFill(psContext, &sRect);
}

//*****************************************************************************
//
// Clears the main screens background.
//
//*****************************************************************************
void
ClearBackground(tContext *psContext)
{
    static const tRectangle sRect =
    {
        BG_MIN_X,
        BG_MIN_Y,
        BG_MAX_X,
        BG_MAX_Y,
    };

    GrRectFill(psContext, &sRect);
}


//*****************************************************************************
//
// Handles when a key is pressed on the keyboard.
//
//*****************************************************************************
void
KeyEvent(tWidget *psWidget, uint32_t ui32Key, uint32_t ui32Event)
{
    switch(ui32Key)
    {
        //
        // Look for a backspace key press.
        //
        case UNICODE_BACKSPACE:
        {
            if(ui32Event == KEYBOARD_EVENT_PRESS)
            {
                if(g_ui32StringIdx != 0)
                {
                    g_ui32StringIdx--;
                    g_pcKeyStr[g_ui32StringIdx] = 0;
                }

                WidgetPaint((tWidget *)&g_sKeyboardText);

                //
                // Save the pixel width of the current string.
                //
                g_i32StringWidth = GrStringWidthGet(&g_sContext, g_pcKeyStr,
                                                    40);
            }
            break;
        }
        //
        // Look for an enter/return key press.  This will exit the keyboard and
        // return to the current active screen.
        //
        case UNICODE_RETURN:
        {
            if(ui32Event == KEYBOARD_EVENT_RELEASE)
            {
                //
                // Get rid of the keyboard widget.
                //
                WidgetRemove(g_sScreens[g_i32ScreenIdx].psWidget);

                //
                // Switch back to the previous screen and add its widget back.
                //
                g_i32ScreenIdx = g_i32ScreenIdx;
                WidgetAdd(WIDGET_ROOT, g_sScreens[g_i32ScreenIdx].psWidget);

                //
                // If returning to the main screen then re-draw the frame to
                // indicate the main screen.
                //
                if(g_i32ScreenIdx == SCREEN_DETAILS)
                {
                    FrameDraw(&g_sContext, "nfc-p2p-demo : Details");
                    WidgetPaint(g_sScreens[g_i32ScreenIdx].psWidget);
                }
                else if(g_i32ScreenIdx == SCREEN_TI)
                {
                    //
                    // Returning to the settings screen.
                    //
                    FrameDraw(&g_sContext, "nfc-p2p-demo : TI");
                    WidgetPaint(g_sScreens[g_i32ScreenIdx].psWidget);
                    AnimateButtons(true);
                    WidgetMessageQueueProcess();
                }
                //
                // Assumed Screen = SCREEN_SUMMARY
                //
                else
                {
                    FrameDraw(&g_sContext, "nfc-p2p-demo : Summary");
                    WidgetPaint(g_sScreens[g_i32ScreenIdx].psWidget);
                }

                //
                // Enable gestures.
                //
                g_sSwipe.bEnable = true;
            }
            break;
        }
        //
        // If the key is not special then update the text string.
        //
        default:
        {
            if(ui32Event == KEYBOARD_EVENT_PRESS)
            {
                //
                // Set the string to the current string to be updated.
                //
                if(g_ui32StringIdx == 0)
                {
                    CanvasTextSet(&g_sKeyboardText, g_pcKeyStr);
                }
                g_pcKeyStr[g_ui32StringIdx] = (char)ui32Key;
                g_ui32StringIdx++;
                g_pcKeyStr[g_ui32StringIdx] = 0;

                WidgetPaint((tWidget *)&g_sKeyboardText);

                //
                // Save the pixel width of the current string.
                //
                g_i32StringWidth = GrStringWidthGet(&g_sContext, g_pcKeyStr,
                                                    40);
            }
            break;
        }
    }
}

//*****************************************************************************
//
// Draws a toggle button.
//
//*****************************************************************************
void
DrawToggle(const tButtonToggle *psButton, bool bOn)
{
    tRectangle sRect;
    int16_t i16X, i16Y;

    //
    // Copy the data out of the bounds of the button.
    //
    sRect = psButton->sRectButton;

    GrContextForegroundSet(&g_sContext, ClrLightGrey);
    GrRectFill(&g_sContext, &psButton->sRectContainer);

    //
    // Copy the data out of the bounds of the button.
    //
    sRect = psButton->sRectButton;

    GrContextForegroundSet(&g_sContext, ClrDarkGray);
    GrRectFill(&g_sContext, &psButton->sRectButton);

    if(bOn)
    {
        sRect.i16XMin += 2;
        sRect.i16YMin += 2;
        sRect.i16XMax -= 15;
        sRect.i16YMax -= 2;
    }
    else
    {
        sRect.i16XMin += 15;
        sRect.i16YMin += 2;
        sRect.i16XMax -= 2;
        sRect.i16YMax -= 2;
    }
    GrContextForegroundSet(&g_sContext, ClrLightGrey);
    GrRectFill(&g_sContext, &sRect);

    GrContextFontSet(&g_sContext, &g_sFontCm16);
    GrContextForegroundSet(&g_sContext, ClrBlack);
    GrContextBackgroundSet(&g_sContext, ClrLightGrey);

    i16X = sRect.i16XMin + ((sRect.i16XMax - sRect.i16XMin) / 2);
    i16Y = sRect.i16YMin + ((sRect.i16YMax - sRect.i16YMin) / 2);

    if(bOn)
    {
        GrStringDrawCentered(&g_sContext, psButton->pcOn, -1, i16X, i16Y,
                             true);
    }
    else
    {
        GrStringDrawCentered(&g_sContext, psButton->pcOff, -1, i16X, i16Y,
                             true);
    }

    if(psButton->pcLabel)
    {
        GrStringDraw(&g_sContext, psButton->pcLabel, -1,
                     psButton->sRectButton.i16XMax + 2,
                     psButton->sRectButton.i16YMin + 6,
                     true);
    }
}

//*****************************************************************************
//
// The interrupt handler for the SysTick interrupt.
//
//*****************************************************************************
void
SysTickIntHandler(void)
{

    //
    // Timeout for the screen saver.
    //
    if(g_ui32ScreenSaver != 0)
    {
        g_ui32ScreenSaver--;
    }

    //
    // Stop updating until the toggle event points have been handled.
    //
    if((g_ui32CursorDelay != 0) &&
       (g_ui32CursorDelay != (KEYBOARD_BLINK_RATE / 2)))
    {
        g_ui32CursorDelay--;
    }
}


//*****************************************************************************
//
// The callback function that is called by the touch screen driver to indicate
// activity on the touch screen.
//
//*****************************************************************************
int32_t
TouchCallback(uint32_t ui32Message, int32_t i32X, int32_t i32Y)
{
    int32_t i32XDiff, i32YDiff;

    //
    // Reset the timeout for the screen saver.
    //
    g_ui32ScreenSaver =  60 * SYSTEM_TICK_S;

    if(g_sSwipe.bEnable)
    {
        switch(ui32Message)
        {
            //
            // The user has just touched the screen.
            //
            case WIDGET_MSG_PTR_DOWN:
            {
                //
                // Save this press location.
                //
                g_sSwipe.i32InitX = i32X;
                g_sSwipe.i32InitY = i32Y;

                //
                // Done handling this message.
                //
                break;
            }

            //
            // The user has moved the touch location on the screen.
            //
            case WIDGET_MSG_PTR_MOVE:
            {
                //
                // Done handling this message.
                //
                break;
            }

            //
            // The user is no longer touching the screen.
            //
            case WIDGET_MSG_PTR_UP:
            {
                //
                // Indicate that no key is being pressed.
                //
                i32XDiff = i32X - g_sSwipe.i32InitX;
                i32YDiff = i32Y - g_sSwipe.i32InitY;

                //
                // Dead zone for just a button press.
                //
                if(((i32XDiff < SWIPE_MIN_DIFF) &&
                    (i32XDiff > -SWIPE_MIN_DIFF)) &&
                   ((i32YDiff < SWIPE_MIN_DIFF) &&
                    (i32YDiff > -SWIPE_MIN_DIFF)))
                {
                    if(g_sButtons.bActive)
                    {
                        //
                        // Reset the delay.
                        //
                        g_sButtons.ui32Delay = 200;

                        if(i32X < 30)
                        {
                            g_sSwipe.eMovement = iSwipeRight;
                        }
                        else if(i32X > 290)
                        {
                            g_sSwipe.eMovement = iSwipeLeft;
                        }
                        else if(i32Y < 40)
                        {
                            g_sSwipe.eMovement = iSwipeDown;
                        }
                        else if(i32Y > 200)
                        {
                            g_sSwipe.eMovement = iSwipeUp;
                        }
                        else
                        {
                            g_sSwipe.eMovement = iSwipeNone;
                        }
                    }
                    else
                    {
                        if(g_i32ScreenIdx == SCREEN_SUMMARY)
                        {
                            AnimateButtons(true);
                        }
                        if(g_i32ScreenIdx == SCREEN_DETAILS)
                        {
                            AnimateButtons(true);
                        }
                    }
                    break;
                }

                //
                // If Y movement dominates then this is an up/down motion.
                //
                if(i32YDiff/i32XDiff)
                {
                    if(i32YDiff < 0)
                    {
                        g_sSwipe.eMovement = iSwipeUp;
                    }
                    else
                    {
                        g_sSwipe.eMovement = iSwipeDown;
                    }
                }
                else
                {
                    if(i32XDiff > 0)
                    {
                        g_sSwipe.eMovement = iSwipeRight;
                    }
                    else
                    {
                        g_sSwipe.eMovement = iSwipeLeft;
                    }
                }

                //
                // Done handling this message.
                //
                break;
            }
        }
    }
    WidgetPointerMessage(ui32Message, i32X, i32Y);

    return(0);
}

//*****************************************************************************
//
// Handle the touch screen movements.
//
//*****************************************************************************
void
HandleMovement(void)
{
    uint32_t ui32NewIdx;

    if(g_sSwipe.eMovement != iSwipeNone)
    {
        switch(g_sSwipe.eMovement)
        {
            case iSwipeUp:
            {
                ui32NewIdx = g_sScreens[g_i32ScreenIdx].ui32Up;
                break;
            }
            case iSwipeDown:
            {
                ui32NewIdx = g_sScreens[g_i32ScreenIdx].ui32Down;

                break;
            }
            case iSwipeRight:
            {
                ui32NewIdx = g_sScreens[g_i32ScreenIdx].ui32Left;

                break;
            }
            case iSwipeLeft:
            {
                ui32NewIdx = g_sScreens[g_i32ScreenIdx].ui32Right;

                break;
            }
            default:
            {
                ui32NewIdx = g_i32ScreenIdx;
                break;
            }
        }

        //
        // Check if the panel has changed.
        //
        if(ui32NewIdx != g_i32ScreenIdx)
        {
            //
            // Remove the current widget.
            //
            WidgetRemove(g_sScreens[g_i32ScreenIdx].psWidget);

            WidgetAdd(WIDGET_ROOT, g_sScreens[ui32NewIdx].psWidget);

            g_i32ScreenIdx = ui32NewIdx;

            //
            // Screen switched so disable the overlay buttons.
            //
            ButtonsDisable();

            if(g_i32ScreenIdx == SCREEN_SUMMARY)
            {
                //
                // Update the frame.
                //
                FrameDraw(&g_sContext, "nfc-p2p-demo : Summary");

                //
                // Animate the panel switch.
                //
                AnimatePanel(TI_BLACK);

                AnimateButtons(true);
            }
            else if(g_i32ScreenIdx == SCREEN_DETAILS)
            {
                //
                // Update the frame.
                //
                FrameDraw(&g_sContext, "nfc-p2p-demo : Details");

                //
                // Animate the panel switch.
                //
                AnimatePanel(TI_BLACK);

                AnimateButtons(true);

            }
            else if(g_i32ScreenIdx == SCREEN_TI)
            {
                //
                // Update the frame.
                //
                FrameDraw(&g_sContext, "nfc-p2p-demo : TI");

                //
                // Animate the panel switch.
                //
                AnimatePanel(TI_GRAY);

                //
                // Animate the pull up tab once.
                //
                AnimateButtons(true);
            }
        }

        g_sSwipe.eMovement = iSwipeNone;
    }
}

//*****************************************************************************
//
// Linear scaling of the palette entries from white to normal color.
//
//*****************************************************************************
static uint8_t
PaletteScale(uint32_t ui32Entry, uint32_t ui32Scale)
{
    uint32_t ui32Value;

    ui32Value = ui32Entry + ((0xff - ui32Entry) * ui32Scale)/15;

    return((uint8_t)ui32Value);
}

//*****************************************************************************
//
// Simple Abstraction to initialize the screen and related processes
//
//*****************************************************************************
void
ScreenInit(void)
{
    //
    // Add the compile-time defined widgets to the widget tree.
    //
    WidgetAdd(WIDGET_ROOT, (tWidget *)&g_sSummaryBackground);

    //
    // Start on the main screen.
    //
    g_i32ScreenIdx = SCREEN_SUMMARY;

    WidgetPaint(WIDGET_ROOT);

    //
    // Configure SysTick for a periodic interrupt at 10ms.
    //
    SysTickPeriodSet((g_ui32SysClk / 1000) * SYSTEM_TICK_MS);
    SysTickEnable();
    SysTickIntEnable();

    //
    // Initialize the swipe state.
    //
    g_sSwipe.eMovement = iSwipeNone;

    //
    // Initialize the touch screen driver.
    //
    TouchScreenInit(g_ui32SysClk);
    TouchScreenCallbackSet(TouchCallback);

    //
    // enable swipe detection
    //
    g_sSwipe.bEnable = true;

    //
    // One minute timeout for screen saver.
    //
    g_ui32ScreenSaver = SYSTEM_TICK_S * 60;
}

//*****************************************************************************
//
// Periodic Functions that need to be called. Should go in main loop
//
//*****************************************************************************
void
ScreenPeriodic(void)
{
    //
    // Handle screen movements.
    //
    HandleMovement();

    //
    // Handle button animation.
    //
    AnimateButtons(true);

    //
    // Handle keyboard entry if it is open.
    //
    HandleKeyboard();

    //
    // If nothing has happened for awhile, then move to a new city.
    //
    if(g_ui32ScreenSaver == 0)
    {
        //
        // Reset the timeout for 10s to update the screen more often.
        //
        g_ui32ScreenSaver = 10 * SYSTEM_TICK_S;

        //
        // Trigger a left swipe.
        //
        g_sSwipe.eMovement = iSwipeLeft;
    }

    WidgetMessageQueueProcess();

}

//*****************************************************************************
//
// Screen Refresh Function. Called when NFC data changes,Move to the Sumary
//  Screen
//
//*****************************************************************************
void
ScreenRefresh(void)
{
    //
    // Repaint All relevant info to screen
    //
    if(g_i32ScreenIdx == SCREEN_SUMMARY)
    {
        WidgetPaint((tWidget *)&g_sTag);
        WidgetPaint((tWidget *)&g_sPayloadLine1);
        WidgetPaint((tWidget *)&g_sPayloadLine2);
        WidgetPaint((tWidget *)&g_sPayloadLine3);
        WidgetPaint((tWidget *)&g_sPayloadLine4);
        WidgetPaint((tWidget *)&g_sPayloadLine5);
        WidgetPaint((tWidget *)&g_sPayloadLine6);
        WidgetPaint((tWidget *)&g_sPayloadLine7);
        WidgetPaint((tWidget *)&g_sPayloadLine8);
    }
    else if(g_i32ScreenIdx == SCREEN_DETAILS)
    {
        g_sSwipe.eMovement = iSwipeLeft;
        HandleMovement();
    }
    else if(g_i32ScreenIdx == SCREEN_TI)
    {
        g_sSwipe.eMovement = iSwipeUp;
        HandleMovement();
    }

    return;
}

//*****************************************************************************
//
// Clear all tag related buffers
//
// Note: 60 is a magic number. It happens to be the buffer length for all the
//        on screen buffers. See screen.h.
//
//*****************************************************************************
void
ScreenClear(void)
{
    uint32_t x=0;
    for(x=0;x<60;x++)
    {
        g_pcPayloadLine1[x]=0;
        g_pcPayloadLine2[x]=0;
        g_pcPayloadLine3[x]=0;
        g_pcPayloadLine4[x]=0;
        g_pcPayloadLine5[x]=0;
        g_pcPayloadLine6[x]=0;
        g_pcPayloadLine7[x]=0;
        g_pcPayloadLine8[x]=0;
        g_pcTagType[x]=0;
        g_pcHeaderLine5[x]=0;
        g_pcHeaderLine4[x]=0;
        g_pcHeaderLine3[x]=0;
        g_pcHeaderLine2[x]=0;
        g_pcHeaderLine1[x]=0;
    }
    return;
}

//*****************************************************************************
//
// Send Data to the Screen starting at line index
// default to start at line 2
//
//*****************************************************************************
void
ScreenPayloadWrite(uint8_t * source, uint32_t length, uint32_t index)
{
    uint32_t x=0;

    switch(index)
    {
        case 1:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine1,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine1,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        default:
        case 2:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine2,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine2,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        case 3:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine3,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine3,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        case 4:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine4,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine4,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        case 5:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine5,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine5,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        case 6:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine6,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine6,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        case 7:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine7,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine7,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
                x=x+SCREEN_LINELENGTH;
        }
        case 8:
        {
                if((length-x) < SCREEN_LINELENGTH)
                {
                    ustrncpy(g_pcPayloadLine8,(char *)source+x, length-x);
                    return;
                }
                else
                {
                    ustrncpy(g_pcPayloadLine8,(char *)source+x,
                                SCREEN_LINELENGTH);
                }
        }
    }
        return;
}

//*****************************************************************************
//
// Handler for Echo Button Press, will echo last tag sent
// call function in nfc_p2p_demo.c
//
//*****************************************************************************
extern void SendData(void);
void OnEchoNFCButton(tWidget *psWidget)
{
    SendData();
    return;
}

//*****************************************************************************
//
// Handler for TI Button Press Will send website data
// call function in nfc_p2p_demo.c
//
//*****************************************************************************
extern void SendTIInfo(void);
void OnTINFCButton(tWidget *psWidget)
{
    SendTIInfo();
    return;
}
