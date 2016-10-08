//*****************************************************************************
//
// capsense.c - Capacitive touch example.
//
// Copyright (c) 2011-2016 Texas Instruments Incorporated.  All rights reserved.
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
// This is part of revision 2.1.3.156 of the EK-TM4C123GXL Firmware Package.
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "drivers/CTS_structure.h"
#include "drivers/CTS_Layer.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Capacitive Touch Example (capsense)</h1>
//!
//! An example that works with the 430BOOST-SENSE1 capactive sense
//! BoosterPack, originally designed for the MSP430 LaunchPad.
//!
//! The TM4C123GH6PM does not have the capacitive sense hardware assisted
//! peripheral features of some MSP430 chips.  Therefore it is required that
//! the user install surface mount resistors on the pads provided on the bottom
//! of the capacitive sense BoosterPack.  Resistor values of 200k ohms are are
//! recommended.  Calibration may be required even when using 200k ohm
//! resistors as each capsense booster pack varies.  Calibration is required
//! for resistors other than 200k ohm.
//!
//! See the wiki page for calibration
//! procedure.  http://processors.wiki.ti.com/index.php/tm4c123g-launchpad
//
//*****************************************************************************


//*****************************************************************************
//
// Number of oscillations to use for each capacitive measurement
//
//*****************************************************************************
#define NUM_OSCILLATIONS                    10

//*****************************************************************************
//
// UART codes for talking to the PC GUI.
//
//*****************************************************************************
#define WAKE_UP_UART_CODE                   0xBE
#define WAKE_UP_UART_CODE2                  0xEF
#define SLEEP_MODE_UART_CODE                0xDE
#define SLEEP_MODE_UART_CODE2               0xAD
#define MIDDLE_BUTTON_CODE                  0x80
#define INVALID_CONVERTED_POSITION          0xFD
#define GESTURE_START                       0xFC
#define GESTURE_STOP                        0xFB
#define COUNTER_CLOCKWISE                   1
#define CLOCKWISE                           2
#define GESTURE_POSITION_OFFSET             0x20
#define WHEEL_POSITION_OFFSET               0x30

#define WHEEL_TOUCH_DELAY                   12

//*****************************************************************************
//
// Global variables to hold the current and previous wheel positions
//
//*****************************************************************************
uint32_t g_ui32WheelPosition = ILLEGAL_SLIDER_WHEEL_POSITION;
uint32_t g_ui32PreviousWheelPosition = ILLEGAL_SLIDER_WHEEL_POSITION;

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
    while(1)
    {
        //
        // Hang on runtime error.
        //
    }
}
#endif

//*****************************************************************************
//
// Delay for count milliseconds
//
//*****************************************************************************
void
DelayMs(uint32_t ui32Count)
{
    ROM_SysCtlDelay((ROM_SysCtlClockGet() / 3000) * ui32Count);
}

//*****************************************************************************
//
// Abstraction for LED output.
//
//*****************************************************************************
void
LEDOutput(uint32_t ui32WheelPosition)
{
    //
    // Put all of our pins in a known state where all LEDs are off.
    //
    ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_4, 0);
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, GPIO_PIN_5);
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTB_BASE, GPIO_PIN_4);
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTB_BASE, GPIO_PIN_6);
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTB_BASE, GPIO_PIN_7);

    if((ui32WheelPosition == 0) || (ui32WheelPosition == 8))
    {
        //
        // If the top or bottom button is being pressed, we have no indicator
        // lights, so return without doing anything.
        //
        return;
    }
    else if(ui32WheelPosition < 8)
    {
        //
        // Positions less than 8 correspond to the right side of the wheel,
        // whose LEDs may turn on when PE4 is pulled high.
        //
        ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_4, GPIO_PIN_4);

        //
        // Write a zero to the appropriate GPIO(s) to turn on the light(s)
        // nearest to the current wheel position.
        //
        switch(ui32WheelPosition)
        {
            case 1:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_5);
                ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, 0);
                break;
            }

            case 2:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_5);
                ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, 0);
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_4, 0);
                break;
            }

            case 3:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_4, 0);
                break;
            }

            case 4:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_4, 0);
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_6, 0);
                break;
            }

            case 5:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_6, 0);
                break;
            }

            case 6:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_6, 0);
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_7);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_7, 0);
                break;
            }

            case 7:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_7);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_7, 0);
                break;
            }

            default:
            {
                //
                // We should never get here.
                //
                break;
            }
        }

    }
    else
    {
        //
        // Positions greater than 8 correspond to the left side of the wheel,
        // whose LEDs may turn on when PE4 is pulled low.
        //
        ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_4, 0);

        //
        // Write a one to the appropriate GPIO(s) to turn on the light(s)
        // nearest to the current wheel position.
        //
        switch(ui32WheelPosition)
        {
            case 9:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_7);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_7, GPIO_PIN_7);
                break;
            }

            case 10:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_7);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_7, GPIO_PIN_7);
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_6, GPIO_PIN_6);
                break;
            }

            case 11:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_6, GPIO_PIN_6);
                break;
            }

            case 12:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_6, GPIO_PIN_6);
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_4, GPIO_PIN_4);
                break;
            }

            case 13:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_4, GPIO_PIN_4);
                break;
            }

            case 14:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
                ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_4, GPIO_PIN_4);
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_5);
                ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, GPIO_PIN_5);
                break;
            }

            case 15:
            {
                ROM_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_5);
                ROM_GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, GPIO_PIN_5);
                break;
            }

            default:
            {
                //
                // We should never get here.
                //
                break;
            }
        }
    }
}

//*****************************************************************************
//
// Takes in a wheel position (0 - 15), compares it with the previous wheel
// position, and returns the distance between the two positions around the
// wheel.  The direction of travel (clockwise vs. counterclockwise) is recorded
// in bit 4.  This is the format in which "gestures" are understood by the PC
// GUI.
//
//*****************************************************************************
uint8_t
GetGesture(uint8_t ui32WheelPosition)
{
    uint_fast8_t ui8Gesture;
    uint_fast8_t ui8Direction, ui8PositionDifference;

    //
    // Start with the assumption that we do not have a valid gesture reading.
    //
    ui8Gesture = INVALID_CONVERTED_POSITION;

    //
    // If our wheel position was previously valid, we can start calculating a
    // gesture reading.
    //
    if(g_ui32PreviousWheelPosition != ILLEGAL_SLIDER_WHEEL_POSITION)
    {
        //
        // If our previous wheel position was bigger than our current wheel
        // position...
        //
        if(g_ui32PreviousWheelPosition > ui32WheelPosition)
        {
            //
            // Calculate a positive difference between the two positions.
            //
            ui8PositionDifference = (g_ui32PreviousWheelPosition -
                                     ui32WheelPosition);

            //
            // If our positive difference is less than 8 (half of the wheel
            // positions), we must have moved counterclockwise.
            //
            if(ui8PositionDifference < 8)
            {
                //
                // Our gesture magnitude will be the positive difference
                // calculated above, and the direction will be
                // counterclockwise.
                //
                ui8Gesture = ui8PositionDifference;
                ui8Direction = COUNTER_CLOCKWISE;
            }
            else
            {
                //
                // If our positive difference is more than 8 (which means it
                // spans more than half of the wheel), then we must have
                // actually moved clockwise.  We should convert our difference
                // measurement to measure the opposite direction around the
                // wheel (which will be the shorter direction), and set our
                // direction to clockwise.
                //
                ui8PositionDifference = 16 - ui8PositionDifference;

                //
                // Make sure the new difference is still less than eight
                // though.
                //
                if(ui8PositionDifference < 8)
                {
                    ui8Gesture = ui8PositionDifference;
                    ui8Direction = CLOCKWISE;
                }
            }
        }
        else
        {
            //
            // If we get here, our current wheel position is larger than our
            // previous wheel position.
            //
            ui8PositionDifference = (ui32WheelPosition -
                                     g_ui32PreviousWheelPosition);
            //
            // As before, make sure we're actually going clockwise, and report
            // our calculated difference as the gesture magnitude.
            //
            if(ui8PositionDifference < 8)
            {
                ui8Gesture = ui8PositionDifference;
                ui8Direction = CLOCKWISE;
            }
            else
            {
                //
                // If we're not actually going clockwise, recalculate our
                // magnitude for counterclockwise movement, and report our
                // actual gesture magnitude and direction appropriately.
                //
                ui8PositionDifference = 16 - ui8PositionDifference;

                //
                // Once again, make sure our new difference is still less than
                // eight.
                //
                if(ui8PositionDifference < 8)
                {
                    ui8Gesture = ui8PositionDifference;
                    ui8Direction = COUNTER_CLOCKWISE;
                }
            }
        }
    }

    //
    // If we made it through that without ever reporting a gesture, return an
    // invalid gesture.
    //
    if(ui8Gesture == INVALID_CONVERTED_POSITION)
    {
        return(ui8Gesture);
    }

    //
    // If our direction was determined to be counterclockwise, set the 4th bit
    // in our return value to indicate this direction to the PC.  Otherwise,
    // return the value as is.
    //
    if(ui8Direction == COUNTER_CLOCKWISE)
    {
        return(ui8Gesture |= 0x10);
    }
    else
    {
        return(ui8Gesture);
    }
}

//*****************************************************************************
//
// Main cap-touch example.
//
//*****************************************************************************
int
main(void)
{
    uint8_t ui8CenterButtonTouched;
    uint32_t ui32WheelTouchCounter;
    uint8_t ui8ConvertedWheelPosition;
    uint8_t ui8GestureDetected;
    uint8_t ui8Loop;

    //
    // Enable lazy stacking for interrupt handlers.  This allows floating-point
    // instructions to be used within interrupt handlers, but at the expense of
    // extra stack usage.
    //
    ROM_FPULazyStackingEnable();

    //
    // Set the clocking to run directly from the PLL at 80 MHz.
    //
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | SYSCTL_XTAL_16MHZ |
                       SYSCTL_OSC_MAIN);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);

    //
    // Initialize a few GPIO outputs for the LEDs
    //
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_4);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, GPIO_PIN_5);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_4);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_6);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_7);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_5);

    //
    // Turn on the Center LED
    //
    ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_5, GPIO_PIN_5);

    //
    // Initialize the UART.
    //
    ROM_GPIOPinConfigure(GPIO_PA0_U0RX);
    ROM_GPIOPinConfigure(GPIO_PA1_U0TX);
    ROM_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);
    UARTStdioConfig(0, 9600, ROM_SysCtlClockGet());

    //
    // Configure the pins needed for capacitive touch sensing.  The capsense
    // driver assumes that these pins are already configured, and that they
    // will be accessed through the AHB bus
    //
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOA);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTA_AHB_BASE, GPIO_PIN_2);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTA_AHB_BASE, GPIO_PIN_3);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTA_AHB_BASE, GPIO_PIN_4);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTA_AHB_BASE, GPIO_PIN_6);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTA_AHB_BASE, GPIO_PIN_7);

    //
    // Start up Systick to measure time.  This is also required by the capsense
    // drivers.
    //
    ROM_SysTickPeriodSet(0x00FFFFFF);
    ROM_SysTickEnable();

    //
    // Set the baseline capacitance measurements for our wheel and center
    // button.
    //
    TI_CAPT_Init_Baseline(&g_sSensorWheel);
    TI_CAPT_Update_Baseline(&g_sSensorWheel, 10);
    TI_CAPT_Init_Baseline(&g_sMiddleButton);
    TI_CAPT_Update_Baseline(&g_sMiddleButton, 10);

    //
    // Send the "sleep" code.  The TIVA C-series version of this app doesn't
    // actually sleep, but the MSP430-based GUI it interacts with expects to
    // see this code on startup, so we will provide it here.
    //
    UARTprintf("\0xDE\0xAD");

    //
    // Send the "awake" code.
    //
    UARTprintf("\0xBE\0xEF");

    //
    // Perform an LED startup sequence.
    //
    for(ui8Loop = 0; ui8Loop < 16; ui8Loop++)
    {
        LEDOutput(ui8Loop);
        DelayMs(10);
    }

    LEDOutput(0);


    //
    // Assume that the center button starts off "untouched", the wheel has no
    // position, and no gestures are in progress.
    //
    ui8CenterButtonTouched = 0;
    ui8ConvertedWheelPosition = INVALID_CONVERTED_POSITION;
    ui8GestureDetected = 0;

    //
    // This "wheel counter" gets incremented when a button on the wheel is held
    // down.  Every time it hits the value of WHEEL_TOUCH_DELAY, the device
    // sends a signal to the GUI reporting another button press.  We want this
    // delay normally, but we'll set the counter to just below the threshold
    // for now, so we'll avoid any percieved lag between the initial button
    // press and the first position report to the GUI.
    //
    ui32WheelTouchCounter = WHEEL_TOUCH_DELAY - 1;

    //
    // Begin the main capsense loop.
    //
    while(1)
    {
        //
        // Start by taking a fresh measurement from the wheel.  If it remains
        // set to ILLEGAL_SLIDER_WHEEL_POSITION, we will know it has not been
        // touched at all.  Otherwise we will have an updated position.
        //
        g_ui32WheelPosition = ILLEGAL_SLIDER_WHEEL_POSITION;
        g_ui32WheelPosition = TI_CAPT_Wheel(&g_sSensorWheel);

        //
        // If we registered a touch somewhere on the wheel, we will need to
        // figure out how to report that touch back to the GUI on the PC.
        //
        if(g_ui32WheelPosition != ILLEGAL_SLIDER_WHEEL_POSITION)
        {
            //
            // First, make sure we're not reporting center button touches while
            // the wheel is active.
            //
            ui8CenterButtonTouched = 0;

            //
            // We need to do a quick formatting change on the wheel position.
            // The "zero" postion as reported by the driver is about 40 degrees
            // off from "up" on the physical wheel.  We'll do that correction
            // here.
            //
            if(g_ui32WheelPosition < 8)
            {
                g_ui32WheelPosition += 64 - 8;
            }
            else
            {
                g_ui32WheelPosition -= 8;
            }

            //
            // We also need to reduce the effective number of positions on the
            // wheel.  The driver reports a wheel position from zero to
            // sixty-three, but the GUI only recognizes positions from zero to
            // sixteen.  Dividing our position by four accomplishes the
            // necessary conversion.
            //
            g_ui32WheelPosition = g_ui32WheelPosition >> 2;

            //
            // Now that we have a properly formatted wheel position, we will
            // use the GetGesture function to determine whether the user has
            // been sliding their finger around the wheel.  If so, this function
            // will return the magnitude and direction of the slide (Check the
            // function description for an example of how this is formated).
            // Otherwise, we will get back a zero.
            //
            ui8ConvertedWheelPosition = GetGesture(g_ui32WheelPosition);

            //
            // If the magnitude of our slide was one wheel position (of
            // sixteen) or less, don't register it.  This prevents excessive
            // reporting of toggles between two adjacent wheel positions.
            //
            if((ui8GestureDetected == 0) &&
               ((ui8ConvertedWheelPosition <= 1) ||
                (ui8ConvertedWheelPosition == 0x11) ||
                (ui8ConvertedWheelPosition == 0x10)))
            {
                //
                // If we obtained a valid wheel position last time we ran this
                // loop, keep our wheel position set to that instead of
                // updating it.  This prevents a mismatch between our recorded
                // absolute position and our recorded swipe magnitude.
                //
                if(g_ui32PreviousWheelPosition !=
                   ILLEGAL_SLIDER_WHEEL_POSITION)
                {
                    g_ui32WheelPosition = g_ui32PreviousWheelPosition;
                }

                //
                // Set the swipe magnitude to zero.
                //
                ui8ConvertedWheelPosition = 0;
            }

            //
            // We've made all of the position adjustments we're going to make,
            // so turn on LEDs to indicate that we've detected a finger on the
            // wheel.
            //
            LEDOutput(g_ui32WheelPosition);

            //
            // If the (adjusted) magnitude of the swipe we detected earlier is
            // valid and non-zero, we should alert the GUI that a gesture is
            // occurring.
            //
            if((ui8ConvertedWheelPosition != 0) &&
               (ui8ConvertedWheelPosition != 16) &&
               (ui8ConvertedWheelPosition != INVALID_CONVERTED_POSITION))
            {
                //
                // If this is a new gesture, we will need to send the gesture
                // start code.
                //
                if(ui8GestureDetected == 0)
                {
                    //
                    // Remember that we've started a gesture.
                    //
                    ui8GestureDetected = 1;

                    //
                    // Transmit gesture start status update & position via UART
                    // to PC.
                    //
                    ROM_UARTCharPut(UART0_BASE, GESTURE_START);
                    ROM_UARTCharPut(UART0_BASE, (g_ui32PreviousWheelPosition +
                                                 GESTURE_POSITION_OFFSET));
                }

                //
                // Transmit gesture & position via UART to PC
                //
                ROM_UARTCharPut(UART0_BASE, ui8ConvertedWheelPosition);
                ROM_UARTCharPut(UART0_BASE, (g_ui32WheelPosition +
                                             GESTURE_POSITION_OFFSET));
            }
            else
            {
                //
                // If we get here, the wheel has been touched, but there hasn't
                // been any sliding recently.  If there hasn't been any sliding
                // AT ALL, then this is a "press" event, and we need to start
                // sending press-style updates to the PC
                //
                if(ui8GestureDetected == 0)
                {
                    //
                    // Increment our wheel counter.
                    //
                    ui32WheelTouchCounter = ui32WheelTouchCounter + 1;

                    //
                    // If the user's finger is still in the same place...
                    //
                    if(ui32WheelTouchCounter >= WHEEL_TOUCH_DELAY)
                    {
                        //
                        // Transmit wheel position (twice) via UART to PC.
                        //
                        ui32WheelTouchCounter = 0;
                        ROM_UARTCharPut(UART0_BASE, (g_ui32WheelPosition +
                                                     WHEEL_POSITION_OFFSET));
                        ROM_UARTCharPut(UART0_BASE, (g_ui32WheelPosition +
                                                     WHEEL_POSITION_OFFSET));
                    }
                }
                else
                {
                    //
                    // We've received a slide input somewhat recently, but not
                    // during this loop instance.  This most likely means that
                    // the user started a gesture, but is currently just
                    // holding their finger in one spot.  This isn't really a
                    // "press" event, so there isn't anything to report.  We
                    // should, however, make sure the touch counter is primed
                    // for future press events.
                    //
                    ui32WheelTouchCounter = WHEEL_TOUCH_DELAY - 1;
                }
            }

            //
            // Regardless of all pressing, sliding, reporting, and LED events
            // that may have occurred, we need to record our measured
            // (adjusted) wheel position for reference for the next pass
            // through the loop.
            //
            g_ui32PreviousWheelPosition = g_ui32WheelPosition;
        }
        else
        {
            //
            // If we get here, there were no touches recorded on the slider
            // wheel.  We should check our middle button to see if it has been
            // pressed, and clean up our recorded state to prepare for future
            // possible wheel-touch events.
            //
            if(TI_CAPT_Button(&g_sMiddleButton))
            {
                //
                // The middle button is currently being touched.  If this is a
                // new touch event, we need to report it to the PC and also
                // toggle our center LED.
                //
                if(ui8CenterButtonTouched == 0)
                {
                    //
                    // Transmit center button code (twice) via UART to PC.
                    //
                    ROM_UARTCharPut(UART0_BASE, MIDDLE_BUTTON_CODE);
                    ROM_UARTCharPut(UART0_BASE, MIDDLE_BUTTON_CODE);

                    //
                    // Record that the center button was touched.
                    //
                    ui8CenterButtonTouched = 1;

                    //
                    // Toggle the center LED.
                    //
                    if(ROM_GPIOPinRead(GPIO_PORTB_BASE, GPIO_PIN_5))
                    {
                        ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_5, 0);
                    }
                    else
                    {
                        ROM_GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_5,
                                         GPIO_PIN_5);
                    }
                }
            }
            else
            {
                //
                // No touch was registered at all (Not wheel or center button).
                // Set our center button state to "untouched", and perform any
                // necessary cleanup related to the wheel.
                //
                ui8CenterButtonTouched = 0;

                //
                // If there haven't been any gestures recently...
                //
                if((ui8ConvertedWheelPosition == INVALID_CONVERTED_POSITION) ||
                   (ui8GestureDetected == 0))
                {
                    //
                    // ... but we do have a valid "previous" wheel position...
                    //
                    if(g_ui32PreviousWheelPosition !=
                       ILLEGAL_SLIDER_WHEEL_POSITION)
                    {
                        //
                        // ... then we probably just had a button-press event
                        // on the wheel.  Send our position data to the computer
                        // one last time to make sure it was recorded.
                        //
                        ROM_UARTCharPut(UART0_BASE,
                                        g_ui32PreviousWheelPosition +
                                        WHEEL_POSITION_OFFSET);
                        ROM_UARTCharPut(UART0_BASE,
                                        g_ui32PreviousWheelPosition +
                                        WHEEL_POSITION_OFFSET);

                        //
                        // Also, reset our wheel touch counter, so we're ready
                        // for new button press events.
                        //
                        ui32WheelTouchCounter = WHEEL_TOUCH_DELAY - 1;
                    }
                }

                if(ui8GestureDetected == 1)
                {
                    //
                    // If we were in the middle of a gesture, a "no touch"
                    // event constitutes a "release".  We need to signal this
                    // to the GUI.
                    //
                    ROM_UARTCharPut(UART0_BASE, GESTURE_STOP);
                    ROM_UARTCharPut(UART0_BASE, GESTURE_STOP);
                }
            }

            //
            // Reset all touch conditions and turn off LEDs
            //
            LEDOutput(0);
            g_ui32PreviousWheelPosition= ILLEGAL_SLIDER_WHEEL_POSITION;
            ui8ConvertedWheelPosition = INVALID_CONVERTED_POSITION;
            ui8GestureDetected = 0;
        }

        //
        // Option: Add delay/sleep cycle here to reduce active duty cycle.
        // This lowers power consumption but sacrifices wheel responsiveness.
        // Additional timing refinement must be taken into consideration when
        // interfacing with PC applications GUI to retain proper communication
        // protocol.
        //
        DelayMs(50);
    }
}
