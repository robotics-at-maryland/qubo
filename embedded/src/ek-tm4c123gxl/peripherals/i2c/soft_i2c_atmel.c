//*****************************************************************************
//
// soft_i2c_atmel.c - Software I2C master example.
//
// Copyright (c) 2010-2016 Texas Instruments Incorporated.  All rights reserved.
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

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/uart.h"
#include "utils/softi2c.h"
#include "utils/uartstdio.h"

//*****************************************************************************
//
//! \addtogroup i2c_examples_list
//! <h1>SoftI2C AT24C08A EEPROM (soft_i2c_atmel)</h1>
//!
//! This example shows how to configure the SoftI2C module to read and write an
//! Atmel AT24C08A EEPROM.  A pattern is written into the first 16 bytes of the
//! EEPROM and then read back.
//!
//! This example uses the following peripherals and I/O signals.  You must
//! review these and change as needed for your own board:
//! - Timer0 peripheral (for the SoftI2C timer)
//! - GPIO Port B peripheral (for SoftI2C pins)
//! - PB2 (for SCL)
//! - PB3 (for SDA)
//!
//! The following UART signals are configured only for displaying console
//! messages for this example.  These are not required for operation of I2C.
//! - UART0 peripheral
//! - GPIO Port A peripheral (for UART0 pins)
//! - UART0RX - PA0
//! - UART0TX - PA1
//!
//! This example uses the following interrupt handlers.  To use this example
//! in your own application, you must add these interrupt handlers to your
//! vector table.
//! - INT_TIMER0A - Timer0AIntHandler
//
//*****************************************************************************

//*****************************************************************************
//
// The I2C slave address of the AT24C08A EEPROM device.  This address is based
// on the A2 pin of the AT24C08A being pulled high on the board.
//
//*****************************************************************************
#define SLAVE_ADDR              0x54

//*****************************************************************************
//
// The states in the interrupt handler state machine.
//
//*****************************************************************************
#define STATE_IDLE              0
#define STATE_WRITE_NEXT        1
#define STATE_WRITE_FINAL       2
#define STATE_WAIT_ACK          3
#define STATE_SEND_ACK          4
#define STATE_READ_ONE          5
#define STATE_READ_FIRST        6
#define STATE_READ_NEXT         7
#define STATE_READ_FINAL        8
#define STATE_READ_WAIT         9

//*****************************************************************************
//
// The state of the SoftI2C module.
//
//*****************************************************************************
static tSoftI2C g_sI2C;

//*****************************************************************************
//
// The variables that track the data to be transmitted or received.
//
//*****************************************************************************
static uint8_t *g_pui8Data = 0;
static uint32_t g_ui32Count = 0;

//*****************************************************************************
//
// The current state of the interrupt handler state machine.
//
//*****************************************************************************
static volatile uint32_t g_ui32State = STATE_IDLE;

//*****************************************************************************
//
// This function sets up UART0 to be used for a console to display information
// as the example is running.
//
//*****************************************************************************
void
InitConsole(void)
{
    //
    // Enable GPIO port A which is used for UART0 pins.
    // TODO: change this to whichever GPIO port you are using.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    //
    // Configure the pin muxing for UART0 functions on port A0 and A1.
    // This step is not necessary if your part does not support pin muxing.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);

    //
    // Enable UART0 so that we can configure the clock.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Use the internal 16MHz oscillator as the UART clock source.
    //
    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    //
    // Select the alternate (UART) function for these pins.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, 16000000);
}

//*****************************************************************************
//
// The callback function for the SoftI2C module.
//
//*****************************************************************************
void
SoftI2CCallback(void)
{
    //
    // Clear the SoftI2C interrupt.
    //
    SoftI2CIntClear(&g_sI2C);

    //
    // Determine what to do based on the current state.
    //
    switch(g_ui32State)
    {
        //
        // The idle state.
        //
        case STATE_IDLE:
        {
            //
            // There is nothing to be done.
            //
            break;
        }

        //
        // The state for the middle of a burst write.
        //
        case STATE_WRITE_NEXT:
        {
            //
            // Write the next data byte.
            //
            SoftI2CDataPut(&g_sI2C, *g_pui8Data++);
            g_ui32Count--;

            //
            // Continue the burst write.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_BURST_SEND_CONT);

            //
            // If there is one byte left, set the next state to the final write
            // state.
            //
            if(g_ui32Count == 1)
            {
                g_ui32State = STATE_WRITE_FINAL;
            }

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the final write of a burst sequence.
        //
        case STATE_WRITE_FINAL:
        {
            //
            // Write the final data byte.
            //
            SoftI2CDataPut(&g_sI2C, *g_pui8Data++);
            g_ui32Count--;

            //
            // Finish the burst write.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_BURST_SEND_FINISH);

            //
            // The next state is to wait for the burst write to complete.
            //
            g_ui32State = STATE_SEND_ACK;

            //
            // This state is done.
            //
            break;
        }

        //
        // Wait for an ACK on the read after a write.
        //
        case STATE_WAIT_ACK:
        {
            //
            // See if there was an error on the previously issued read.
            //
            if(SoftI2CErr(&g_sI2C) == SOFTI2C_ERR_NONE)
            {
                //
                // Read the byte received.
                //
                SoftI2CDataGet(&g_sI2C);

                //
                // There was no error, so the state machine is now idle.
                //
                g_ui32State = STATE_IDLE;

                //
                // This state is done.
                //
                break;
            }

            //
            // Fall through to STATE_SEND_ACK.
            //
        }

        //
        // Send a read request, looking for the ACK to indicate that the write
        // is done.
        //
        case STATE_SEND_ACK:
        {
            //
            // Put the I2C master into receive mode.
            //
            SoftI2CSlaveAddrSet(&g_sI2C, SLAVE_ADDR, true);

            //
            // Perform a single byte read.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_SINGLE_RECEIVE);

            //
            // The next state is the wait for the ack.
            //
            g_ui32State = STATE_WAIT_ACK;

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for a single byte read.
        //
        case STATE_READ_ONE:
        {
            //
            // Put the SoftI2C module into receive mode.
            //
            SoftI2CSlaveAddrSet(&g_sI2C, SLAVE_ADDR, true);

            //
            // Perform a single byte read.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_SINGLE_RECEIVE);

            //
            // The next state is the wait for final read state.
            //
            g_ui32State = STATE_READ_WAIT;

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the start of a burst read.
        //
        case STATE_READ_FIRST:
        {
            //
            // Put the SoftI2C module into receive mode.
            //
            SoftI2CSlaveAddrSet(&g_sI2C, SLAVE_ADDR, true);

            //
            // Start the burst receive.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_BURST_RECEIVE_START);

            //
            // The next state is the middle of the burst read.
            //
            g_ui32State = STATE_READ_NEXT;

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the middle of a burst read.
        //
        case STATE_READ_NEXT:
        {
            //
            // Read the received character.
            //
            *g_pui8Data++ = SoftI2CDataGet(&g_sI2C);
            g_ui32Count--;

            //
            // Continue the burst read.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_BURST_RECEIVE_CONT);

            //
            // If there are two characters left to be read, make the next
            // state be the end of burst read state.
            //
            if(g_ui32Count == 2)
            {
                g_ui32State = STATE_READ_FINAL;
            }

            //
            // This state is done.
            //
            break;
        }

        //
        // The state for the end of a burst read.
        //
        case STATE_READ_FINAL:
        {
            //
            // Read the received character.
            //
            *g_pui8Data++ = SoftI2CDataGet(&g_sI2C);
            g_ui32Count--;

            //
            // Finish the burst read.
            //
            SoftI2CControl(&g_sI2C, SOFTI2C_CMD_BURST_RECEIVE_FINISH);

            //
            // The next state is the wait for final read state.
            //
            g_ui32State = STATE_READ_WAIT;

            //
            // This state is done.
            //
            break;
        }

        //
        // This state is for the final read of a single or burst read.
        //
        case STATE_READ_WAIT:
        {
            //
            // Read the received character.
            //
            *g_pui8Data++ = SoftI2CDataGet(&g_sI2C);
            g_ui32Count--;

            //
            // The state machine is now idle.
            //
            g_ui32State = STATE_IDLE;

            //
            // This state is done.
            //
            break;
        }
    }
}

//*****************************************************************************
//
// Write to the Atmel device.
//
//*****************************************************************************
void
AtmelWrite(uint8_t *pui8Data, uint32_t ui32Offset, uint32_t ui32Count)
{
    //
    // Save the data buffer to be written.
    //
    g_pui8Data = pui8Data;
    g_ui32Count = ui32Count;

    //
    // Set the next state of the callback state machine based on the number of
    // bytes to write.
    //
    if(ui32Count != 1)
    {
        g_ui32State = STATE_WRITE_NEXT;
    }
    else
    {
        g_ui32State = STATE_WRITE_FINAL;
    }

    //
    // Set the slave address and setup for a transmit operation.
    //
    SoftI2CSlaveAddrSet(&g_sI2C, SLAVE_ADDR | (ui32Offset >> 8), false);

    //
    // Write the address to be written as the first data byte.
    //
    SoftI2CDataPut(&g_sI2C, ui32Offset);

    //
    // Start the burst cycle, writing the address as the first byte.
    //
    SoftI2CControl(&g_sI2C, SOFTI2C_CMD_BURST_SEND_START);

    //
    // Wait until the SoftI2C callback state machine is idle.
    //
    while(g_ui32State != STATE_IDLE)
    {
    }
}

//*****************************************************************************
//
// Read from the Atmel device.
//
//*****************************************************************************
void
AtmelRead(uint8_t *pui8Data, uint32_t ui32Offset, uint32_t ui32Count)
{
    //
    // Save the data buffer to be read.
    //
    g_pui8Data = pui8Data;
    g_ui32Count = ui32Count;

    //
    // Set the next state of the callback state machine based on the number of
    // bytes to read.
    //
    if(ui32Count == 1)
    {
        g_ui32State = STATE_READ_ONE;
    }
    else
    {
        g_ui32State = STATE_READ_FIRST;
    }

    //
    // Start with a dummy write to get the address set in the EEPROM.
    //
    SoftI2CSlaveAddrSet(&g_sI2C, SLAVE_ADDR | (ui32Offset >> 8), false);

    //
    // Write the address to be written as the first data byte.
    //
    SoftI2CDataPut(&g_sI2C, ui32Offset);

    //
    // Perform a single send, writing the address as the only byte.
    //
    SoftI2CControl(&g_sI2C, SOFTI2C_CMD_SINGLE_SEND);

    //
    // Wait until the SoftI2C callback state machine is idle.
    //
    while(g_ui32State != STATE_IDLE)
    {
    }
}

//*****************************************************************************
//
// This is the interrupt handler for the Timer0A interrupt.
//
//*****************************************************************************
void
Timer0AIntHandler(void)
{
    //
    // Clear the timer interrupt.
    // TODO: change this to whichever timer you are using.
    //
    TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

    //
    // Call the SoftI2C tick function.
    //
    SoftI2CTimerTick(&g_sI2C);
}

//*****************************************************************************
//
// This example demonstrates the use of the SoftI2C module to read and write an
// Atmel AT24C08A EEPROM.
//
//*****************************************************************************
int
main(void)
{
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    uint32_t ui32SysClock;
#endif
    uint8_t pui8Data[16];
    uint32_t ui32Idx;

    //
    // Set the clocking to run directly from the external crystal/oscillator.
    // TODO: The SYSCTL_XTAL_ value must be changed to match the value of the
    // crystal on your board.
    //

#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    ui32SysClock = SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                       SYSCTL_OSC_MAIN |
                                       SYSCTL_USE_OSC), 25000000);
#else
    SysCtlClockSet(SYSCTL_SYSDIV_1 | SYSCTL_USE_OSC | SYSCTL_OSC_MAIN |
                   SYSCTL_XTAL_16MHZ);
#endif

    //
    // For this example, PortB[3:2] are used for the SoftI2C pins.  GPIO port B
    // needs to be enabled so these pins can be used.
    // TODO: change this to whichever GPIO port(s) you are using.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);

    //
    // For this example, Timer0 is used for the SoftI2C time base.  This timer
    // needs to be enabled before it can be used.
    // TODO: change this to whichever timer you are using.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);

    //
    // Configure the appropriate pins to be I2C instead of GPIO.
    // TODO: change this to select the port/pin you are using.
    //
    GPIOPinTypeI2C(GPIO_PORTB_BASE, GPIO_PIN_2 | GPIO_PIN_3);

    //
    // Initialize the SoftI2C module, including the assignment of GPIO pins.
    // TODO: change this to whichever GPIO pins you are using.
    //
    memset(&g_sI2C, 0, sizeof(g_sI2C));
    SoftI2CCallbackSet(&g_sI2C, SoftI2CCallback);
    SoftI2CSCLGPIOSet(&g_sI2C, GPIO_PORTB_BASE, GPIO_PIN_2);
    SoftI2CSDAGPIOSet(&g_sI2C, GPIO_PORTB_BASE, GPIO_PIN_3);
    SoftI2CInit(&g_sI2C);

    //
    // Enable the SoftI2C interrupt.
    //
    SoftI2CIntEnable(&g_sI2C);

    //
    // Configure the timer to generate an interrupt at a rate of 40 KHz.  This
    // will result in a I2C rate of 10 KHz.
    // TODO: change this to whichever timer you are using.
    // TODO: change this to whichever I2C rate you require.
    //
    TimerConfigure(TIMER0_BASE, TIMER_CFG_PERIODIC);
#if defined(TARGET_IS_TM4C129_RA0) ||                                         \
    defined(TARGET_IS_TM4C129_RA1) ||                                         \
    defined(TARGET_IS_TM4C129_RA2)
    TimerLoadSet(TIMER0_BASE, TIMER_A, ui32SysClock / 40000);
#else
    TimerLoadSet(TIMER0_BASE, TIMER_A, SysCtlClockGet() / 40000);
#endif
    TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
    TimerEnable(TIMER0_BASE, TIMER_A);

    //
    // Enable the timer interrupt.
    // TODO: change this to whichever timer interrupt you are using.
    //
    IntEnable(INT_TIMER0A);

    //
    // Set up the serial console to use for displaying messages.  This is
    // just for this example program and is not needed for SoftI2C operation.
    //
    InitConsole();

    //
    // Display the example setup on the console.
    //
    UARTprintf("SoftI2C Atmel AT24C08A example\n");

    //
    // Write a data=address pattern into the first 16 bytes of the Atmel
    // device.
    //
    UARTprintf("Write:");
    for(ui32Idx = 0; ui32Idx < 16; ui32Idx++)
    {
        pui8Data[ui32Idx] = ui32Idx;
        UARTprintf(" %02x", pui8Data[ui32Idx]);
    }
    UARTprintf("\n");
    AtmelWrite(pui8Data, 0, 16);

    //
    // Read back the first 16 bytes of the Atmel device.
    //
    AtmelRead(pui8Data, 0, 16);
    UARTprintf("Read :");
    for(ui32Idx = 0; ui32Idx < 16; ui32Idx++)
    {
        UARTprintf(" %02x", pui8Data[ui32Idx]);
    }
    UARTprintf("\n");

    //
    // Tell the user that the test is done.
    //
    UARTprintf("Done.\n\n");

    //
    // Return no errors.
    //
    return(0);
}
