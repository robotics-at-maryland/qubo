//*****************************************************************************
//
// command_task.h - Virtual COM Port Task manage messages to and from terminal.
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

#ifndef __COMMAND_TASK_H__
#define __COMMAND_TASK_H__

//*****************************************************************************
//
// The stack size for the COMMAND task.
//
//*****************************************************************************
#define COMMAND_TASK_STACK_SIZE        1024         // Stack size in words

//*****************************************************************************
//
// Period in milliseconds to determine time between how often run the task.
// This determines how often we check for the carriage return which indicates
// we need to process a command.
//
//*****************************************************************************
#define COMMAND_TASK_PERIOD_MS         50        // periodic rate of the task

//*****************************************************************************
//
// A handle by which this task and others can refer to this task.
//
//*****************************************************************************
extern xTaskHandle g_xCommandHandle;

//*****************************************************************************
//
// A handle for the UART semaphore. Only used when more than one message must
// be kept in order without other tasks injecting messages in between.
//
//*****************************************************************************
extern xSemaphoreHandle g_xUARTSemaphore;

//*****************************************************************************
//
// Prototypes for the command task.
//
//*****************************************************************************
extern uint32_t CommandTaskInit(void);

//*****************************************************************************
//
// Forward declarations for command-line operations.
//
//*****************************************************************************
extern int Cmd_help(int argc, char *argv[]);
extern int Cmd_stats(int argc, char *argv[]);
extern int Cmd_activate(int argc, char *argv[]);
extern int Cmd_clear(int argc, char *argv[]);
extern int Cmd_sync(int argc, char *argv[]);
extern int Cmd_connect(int argc, char *argv[]);
extern int Cmd_proxy(int argc, char *argv[]);

#endif // __COMMAND_TASK_H__
