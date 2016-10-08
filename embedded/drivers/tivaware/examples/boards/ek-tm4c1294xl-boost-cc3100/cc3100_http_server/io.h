//*****************************************************************************
//
// io.h - Prototypes for I/O routines for the enet_io example.
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
//
//*****************************************************************************

#ifndef __IO_H__
#define __IO_H__

#ifdef __cplusplus
extern "C"
{
#endif

struct fsdata_file {
  const struct fsdata_file *next;
  const unsigned char *name;
  const unsigned char *data;
  int len;
  uint8_t http_header_included;
};

//*****************************************************************************
//
// Exported global variables.
//
//*****************************************************************************
extern volatile unsigned long g_ulAnimSpeed;

//*****************************************************************************
//
// Exported function prototypes.
//
//*****************************************************************************
void io_set_timer(uint32_t ui32SpeedPercent);
void io_init(void);
void io_set_led(bool bOn);
void io_set_animation_speed_string(char *pcBuf);
void io_set_animation_speed(uint32_t ui32SpeedPercent);
int32_t io_is_led_on(void);

#ifdef __cplusplus
}
#endif

#endif // __IO_H__
