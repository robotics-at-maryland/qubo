//*****************************************************************************
//
// screen.h - Touch Screen movements / Canvases for NFC P2P Demo.
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

#ifndef __NFC_P2P_SCREEN__
#define __NFC_P2P_SCREEN__

//*****************************************************************************
//
// Hex Color Deffinitions for official Texas Instruments Colors
//
//*****************************************************************************
#define TI_GRAY         0x00999999
#define TI_RED          0x00FF0000
#define TI_BLACK        0x00000000
#define TI_WHITE        0x00FFFFFF
#define TI_PANTONE1807  0x00990000
#define TI_PANTONE5473  0x00115566
#define TI_PANTONE321   0x00118899
#define TI_PANTONE3125  0x004ABED4
#define TI_PANTONE5455  0x00DCDCDE

//*****************************************************************************
//
// Number of characters per line in the Payload section on Screen
// (must be less than the length of the g_pcPayloadLineX[] arrays)
//
//*****************************************************************************
#define SCREEN_LINELENGTH 45

//*****************************************************************************
//
// Deffinitions for functions
//
//*****************************************************************************
extern uint32_t g_ui32SysClk;
extern void ScreenInit(void);
extern void ScreenPeriodic(void);
extern void ScreenRefresh(void);
extern void ScreenClear(void);
extern void ScreenPayloadWrite(uint8_t *source, uint32_t length,uint32_t index);
extern tContext g_sContext;
extern char g_pcHeaderLine1[60];
extern char g_pcHeaderLine2[60];
extern char g_pcHeaderLine3[60];
extern char g_pcHeaderLine4[60];
extern char g_pcHeaderLine5[60];
extern char g_pcPayloadLine1[60];
extern char g_pcPayloadLine2[60];
extern char g_pcPayloadLine3[60];
extern char g_pcPayloadLine4[60];
extern char g_pcPayloadLine5[60];
extern char g_pcPayloadLine6[60];
extern char g_pcPayloadLine7[60];
extern char g_pcPayloadLine8[60];
extern char g_pcTagType[60];


#endif //__NFC_P2P_SCREEN__
