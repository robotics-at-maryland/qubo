//*****************************************************************************
//
// usb_dev_keyboard.c - Main routines for the keyboard example.
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
// This is part of revision 2.1.3.156 of the EK-TM4C123GXL Firmware Package.
//
//*****************************************************************************

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"
#include "inc/hw_sysctl.h"
#include "driverlib/adc.h"
#include "driverlib/debug.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "usblib/device/usbdevice.h"
#include "usblib/device/usbdcomp.h"
#include "usblib/device/usbdhid.h"
#include "usblib/device/usbdhidgamepad.h"
#include "usb_cgamepad_structs.h"
#include "utils/uartstdio.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>USB HID Composite Gamepad (usb_dev_cgamepad)</h1>
//!
//! This example application enables the evaluation board to act as a dual USB
//! game pad device supported using the Human Interface Device class. The
//! mapping of the analog pin to gamepad axis and GPIO to button inputs are
//! listed below.
//!
//! Analog Pin Mapping:
//!
//! - Gamepad 1 X  Axis - PE2/AIN1
//! - Gamepad 1 Y  Axis - PE1/AIN2
//! - Gamepad 1 Z  Axis - PD3/AIN4
//! - Gamepad 1 Rx Axis - PD2/AIN5
//!
//! - Gamepad 2 X  Axis - PD1/AIN6
//! - Gamepad 2 Y  Axis - PD0/AIN7
//! - Gamepad 2 Z  Axis - PE5/AIN8
//! - Gamepad 2 Ry Axis - PB5/AIN11
//!
//! Button Pin Mapping.
//!
//! - Gamepad 1 Button  1 - PF4
//! - Gamepad 1 Button  2 - PE0
//! - Gamepad 1 Button  3 - PE3
//! - Gamepad 1 Button  4 - PE4
//! - Gamepad 1 Button  5 - PB4
//! - Gamepad 1 Button  6 - PB3
//! - Gamepad 1 Button  7 - PB2
//! - Gamepad 1 Button  8 - PB0
//! - Gamepad 1 Button  9 - PB1
//! - Gamepad 1 Button 10 - PA6
//! - Gamepad 1 Button 11 - PA7
//!
//! - Gamepad 2 Button  1 - PF0
//! - Gamepad 2 Button  2 - PC4
//! - Gamepad 2 Button  3 - PC5
//! - Gamepad 2 Button  4 - PC6
//! - Gamepad 2 Button  5 - PC7
//! - Gamepad 2 Button  6 - PD6
//! - Gamepad 2 Button  7 - PD7
//! - Gamepad 2 Button  8 - PA5
//! - Gamepad 2 Button  9 - PA4
//! - Gamepad 2 Button 10 - PA3
//! - Gamepad 2 Button 11 - PA2
//
//*****************************************************************************

#define PORTA_BUTTON_PINS   (GPIO_PIN_2 | GPIO_PIN_3 | GPIO_PIN_4 |           \
                             GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7)
#define PORTB_BUTTON_PINS   (GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 |           \
                             GPIO_PIN_3 | GPIO_PIN_4)
#define PORTC_BUTTON_PINS   (GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7)
#define PORTD_BUTTON_PINS   (GPIO_PIN_6 | GPIO_PIN_7)
#define PORTE_BUTTON_PINS   (GPIO_PIN_0 | GPIO_PIN_3 | GPIO_PIN_4)
#define PORTF_BUTTON_PINS   (GPIO_PIN_0 | GPIO_PIN_4)
#define PORTF_SWITCH_PINS   (GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3)

//*****************************************************************************
//
// The system tick timer period.
//
//*****************************************************************************
#define SYSTICKS_PER_SECOND     100

//*****************************************************************************
//
//! The default packed report structure that is sent to the host.  This should
//! only be used if the report descriptor is not overridden by the application.
//
//*****************************************************************************
typedef struct
{
    uint16_t i16XPos;
    uint16_t i16YPos;
    uint16_t i16RXPos;
    uint16_t i16RYPos;
    uint16_t ui16Buttons;
}
PACKED tCustomReport;

//*****************************************************************************
//
// The memory allocated to hold the composite descriptor that is created by
// the call to USBDCompositeInit().
//
//*****************************************************************************
#define DESCRIPTOR_DATA_SIZE    (COMPOSITE_DHID_SIZE + COMPOSITE_DHID_SIZE)
uint8_t g_pui8DescriptorData[DESCRIPTOR_DATA_SIZE];

//*****************************************************************************
//
// Global system tick counter holds elapsed time since the application started
// expressed in 100ths of a second.
//
//*****************************************************************************
volatile uint32_t g_ui32SysTickCount;

//*****************************************************************************
//
// The number of system ticks to wait for each USB packet to be sent before
// we assume the host has disconnected.  The value 50 equates to half a second.
//
//*****************************************************************************
#define MAX_SEND_DELAY          50

//*****************************************************************************
//
// The HID gamepad report that is returned to the host.
//
//*****************************************************************************
static tCustomReport sReportA, sReportB;

//*****************************************************************************
//
// The HID gamepad polled ADC data for the coordinates.
//
//*****************************************************************************
static uint32_t g_pui32ADCData[8];

//*****************************************************************************
//
// Button state variables.
//
//*****************************************************************************
#define BUTTON_MASK_A           0x0000e7ff
#define BUTTON_MASK_B           0x07ff0000
uint32_t g_ui32ButtonStates;

//*****************************************************************************
//
// This enumeration holds the various states that the keyboard can be in during
// normal operation.
//
//*****************************************************************************
static volatile enum
{
    //
    // Not yet configured.
    //
    eStateNotConfigured,

    //
    // Connected and not waiting on data to be sent.
    //
    eStateIdle,

    //
    // Suspended.
    //
    eStateSuspend,

    //
    // Connected and waiting on data to be sent out only on gamepad A.
    //
    eStateSendingA,

    //
    // Connected and waiting on data to be sent out only on gamepad B.
    //
    eStateSendingB,

    //
    // Connected and waiting on data to be sent out only on gamepad A and B.
    //
    eStateSendingAB,
}
g_iGamepadState;

//*****************************************************************************
//
// The error routine that is called if the driver library encounters an error.
//
//*****************************************************************************
#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
}
#endif

//*****************************************************************************
//
// Handles asynchronous events from the HID keyboard driver.
//
// \param pvCBData is the event callback pointer provided during
// USBDHIDKeyboardInit().  This is a pointer to our keyboard device structure
// (&g_sKeyboardDevice).
// \param ui32Event identifies the event we are being called back for.
// \param ui32MsgData is an event-specific value.
// \param pvMsgData is an event-specific pointer.
//
// This function is called by the HID keyboard driver to inform the application
// of particular asynchronous events related to operation of the keyboard HID
// device.
//
// \return Returns 0 in all cases.
//
//*****************************************************************************
uint32_t
GamepadHandler(void *pvCBData, uint32_t ui32Event, uint32_t ui32MsgData,
               void *pvMsgData)
{
    switch (ui32Event)
    {
        //
        // The host has connected to us and configured the device.
        //
        case USB_EVENT_CONNECTED:
        {
            if((void *)&g_sGamepadDeviceA == pvCBData)
            {
                UARTprintf("USB Device 0 Connected\n");
            }
            else
            {
                UARTprintf("USB Device 1 Connected\n");
            }

            g_iGamepadState = eStateIdle;

            break;
        }

        //
        // The host has disconnected from us.
        //
        case USB_EVENT_DISCONNECTED:
        {
            UARTprintf("USB disconnected\n");

            g_iGamepadState = eStateNotConfigured;

            break;
        }

        //
        // We receive this event every time the host acknowledges transmission
        // of a report. It is used here purely as a way of determining whether
        // the host is still talking to us or not.
        //
        case USB_EVENT_TX_COMPLETE:
        {
            if((tUSBDHIDGamepadDevice *)pvCBData == &g_sGamepadDeviceA)
            {
                if(g_iGamepadState == eStateSendingA)
                {
                    //
                    // Now idle, nothing is sending.
                    //
                    g_iGamepadState = eStateIdle;
                }
                else if(g_iGamepadState == eStateSendingAB)
                {
                    //
                    // Still waiting on gamepad B.
                    //
                    g_iGamepadState = eStateSendingB;
                }
                else
                {
                    //
                    // Should never get here.
                    //
                    ASSERT(1);
                }
            }
            else if((tUSBDHIDGamepadDevice *)pvCBData == &g_sGamepadDeviceB)
            {
                if(g_iGamepadState == eStateSendingB)
                {
                    //
                    // Now idle, nothing is sending.
                    //
                    g_iGamepadState = eStateIdle;
                }
                else if(g_iGamepadState == eStateSendingAB)
                {
                    //
                    // Still waiting on gamepad A.
                    //
                    g_iGamepadState = eStateSendingA;
                }
                else
                {
                    //
                    // Should never get here.
                    //
                    ASSERT(1);
                }
            }
            g_iGamepadState = eStateIdle;

            break;
        }

        //
        // This event indicates that the host has suspended the USB bus.
        //
        case USB_EVENT_SUSPEND:
        {
            //
            // Go to the suspended state.
            //
            g_iGamepadState = eStateSuspend;

            break;
        }

        //
        // This event signals that the host has resumed signaling on the bus.
        //
        case USB_EVENT_RESUME:
        {
            //
            // Go back to the idle state.
            //
            g_iGamepadState = eStateIdle;

            break;
        }

        case USBD_HID_EVENT_GET_REPORT:
        {
            //
            // Return the pointer to the current report.  This call is
            // rarely if ever made, but is required by the USB HID
            // specification.
            //
            *(void **)pvMsgData = (void *)&sReportA;

            break;
        }

        //
        // We ignore all other events.
        //
        default:
        {
            break;
        }
    }

    return(0);
}

//*****************************************************************************
//
// This is the interrupt handler for the SysTick interrupt.  It is used to
// update our local tick count which, in turn, is used to check for transmit
// timeouts.
//
//*****************************************************************************
void
SysTickIntHandler(void)
{
    g_ui32SysTickCount++;
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
    // Use the internal 16MHz oscillator as the UART clock source.
    //
    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, 16000000);
}

//*****************************************************************************
//
// The game pad button press/release state machine.
//
//*****************************************************************************
uint32_t
GamepadButtonsGet(uint32_t *pui32Delta, uint32_t *pui32RawState)
{
    uint32_t ui32Delta;
    static uint32_t ui32SwitchClockA = 0;
    static uint32_t ui32SwitchClockB = 0;
    uint32_t ui32Buttons;
    uint32_t ui32PortA, ui32PortB, ui32PortC, ui32PortD, ui32PortE, ui32PortF;

    //
    // Read the raw state of the push buttons.  Save the raw state
    // (inverting the bit sense) if the caller supplied storage for the
    // raw value.
    //
    ui32PortA = ROM_GPIOPinRead(GPIO_PORTA_AHB_BASE, PORTA_BUTTON_PINS);
    ui32PortB = ROM_GPIOPinRead(GPIO_PORTB_AHB_BASE, PORTB_BUTTON_PINS);
    ui32PortC = ROM_GPIOPinRead(GPIO_PORTC_AHB_BASE, PORTC_BUTTON_PINS);
    ui32PortD = ROM_GPIOPinRead(GPIO_PORTD_AHB_BASE, PORTD_BUTTON_PINS);
    ui32PortE = ROM_GPIOPinRead(GPIO_PORTE_AHB_BASE, PORTE_BUTTON_PINS);
    ui32PortF = ROM_GPIOPinRead(GPIO_PORTF_AHB_BASE, PORTF_BUTTON_PINS |
                                                     PORTF_SWITCH_PINS);

    ui32Buttons = (((ui32PortF >> 4) & 1) |                 // PF4
                   (((ui32PortE >> 0) & 1) << 1) |          // PE0
                   (((ui32PortE >> 3) & 1) << 2) |          // PE3
                   (((ui32PortE >> 4) & 1) << 3) |          // PE4
                   (((ui32PortB >> 4) & 1) << 4) |          // PB4
                   (((ui32PortB >> 3) & 1) << 5) |          // PB3
                   (((ui32PortB >> 2) & 1) << 6) |          // PB2
                   (((ui32PortB >> 0) & 1) << 7) |          // PB0
                   (((ui32PortB >> 1) & 1) << 8) |          // PB1
                   (((ui32PortA >> 6) & 1) << 9) |          // PA6
                   (((ui32PortA >> 7) & 1) << 10) |         // PA7
                   (((ui32PortF >> 1) & 1) << 13) |         // PF1
                   (((ui32PortF >> 2) & 1) << 14) |         // PF2
                   (((ui32PortF >> 3) & 1) << 15) |         // PF3
                   (((ui32PortF >> 0) & 1) << (16 + 0)) |   // PF0
                   (((ui32PortC >> 4) & 1) << (16 + 1)) |   // PC4
                   (((ui32PortC >> 5) & 1) << (16 + 2)) |   // PC5
                   (((ui32PortC >> 6) & 1) << (16 + 3)) |   // PC6
                   (((ui32PortC >> 7) & 1) << (16 + 4)) |   // PC7
                   (((ui32PortD >> 6) & 1) << (16 + 5)) |   // PD6
                   (((ui32PortD >> 7) & 1) << (16 + 6)) |   // PD7
                   (((ui32PortA >> 5) & 1) << (16 + 7)) |   // PA5
                   (((ui32PortA >> 4) & 1) << (16 + 8)) |   // PA4
                   (((ui32PortA >> 3) & 1) << (16 + 9)) |   // PA3
                   (((ui32PortA >> 2) & 1) << (16 + 10)));  // PA2

    if(pui32RawState)
    {
        *pui32RawState = ~(ui32Buttons & (BUTTON_MASK_A | BUTTON_MASK_B));
    }

    //
    // Determine the switches that are at a different state than the debounced
    // state.
    //
    ui32Delta = ui32Buttons ^ g_ui32ButtonStates;

    //
    // Increment the clocks by one.
    //
    ui32SwitchClockA ^= ui32SwitchClockB;
    ui32SwitchClockB = ~ui32SwitchClockB;

    //
    // Reset the clocks corresponding to switches that have not changed state.
    //
    ui32SwitchClockA &= ui32Delta;
    ui32SwitchClockB &= ui32Delta;

    //
    // Get the new debounced switch state.
    //
    g_ui32ButtonStates &= ui32SwitchClockA | ui32SwitchClockB;
    g_ui32ButtonStates |= (~(ui32SwitchClockA | ui32SwitchClockB)) & ui32Buttons;

    //
    // Determine the switches that just changed debounced state.
    //
    ui32Delta ^= (ui32SwitchClockA | ui32SwitchClockB);

    //
    // Store the bit mask for the buttons that have changed for return to
    // caller.
    //
    if(pui32Delta)
    {
        *pui32Delta = ui32Delta & (BUTTON_MASK_A | BUTTON_MASK_B);
    }

    //
    // Return the debounced buttons states to the caller.  Invert the bit
    // sense so that a '1' indicates the button is pressed, which is a
    // sensible way to interpret the return value.
    //
    return(~(g_ui32ButtonStates & (BUTTON_MASK_A | BUTTON_MASK_B)));
}

//*****************************************************************************
//
// Initialize the GPIO inputs used by the game pad device.
//
//*****************************************************************************
void
GamepadButtonsInit(void)
{
    //
    // Enable the GPIO ports to which the pushbuttons are connected.
    //
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOA);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOB);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOC);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOC);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOD);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOE);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOF);

    //
    // Set each of the button GPIO pins as an input with a pull-up.
    //
    ROM_GPIODirModeSet(GPIO_PORTA_AHB_BASE, PORTA_BUTTON_PINS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(GPIO_PORTA_AHB_BASE, PORTA_BUTTON_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPU);

    ROM_GPIODirModeSet(GPIO_PORTB_AHB_BASE, PORTB_BUTTON_PINS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(GPIO_PORTB_AHB_BASE, PORTB_BUTTON_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPU);

    ROM_GPIODirModeSet(GPIO_PORTC_AHB_BASE, PORTC_BUTTON_PINS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(GPIO_PORTC_AHB_BASE, PORTC_BUTTON_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPU);

    HWREG(GPIO_PORTD_AHB_BASE + GPIO_O_LOCK) = GPIO_LOCK_KEY;
    HWREG(GPIO_PORTD_AHB_BASE + GPIO_O_CR) |= GPIO_PIN_7;
    HWREG(GPIO_PORTD_AHB_BASE + GPIO_O_LOCK) = 0;

    ROM_GPIODirModeSet(GPIO_PORTD_AHB_BASE, PORTD_BUTTON_PINS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(GPIO_PORTD_AHB_BASE, PORTD_BUTTON_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPU);

    ROM_GPIODirModeSet(GPIO_PORTE_AHB_BASE, PORTE_BUTTON_PINS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(GPIO_PORTE_AHB_BASE, PORTE_BUTTON_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPU);

    HWREG(GPIO_PORTF_AHB_BASE + GPIO_O_LOCK) = GPIO_LOCK_KEY;
    HWREG(GPIO_PORTF_AHB_BASE + GPIO_O_CR) |= PORTF_BUTTON_PINS | PORTF_SWITCH_PINS;
    HWREG(GPIO_PORTF_AHB_BASE + GPIO_O_LOCK) = 0;

    ROM_GPIODirModeSet(GPIO_PORTF_AHB_BASE,
                       PORTF_BUTTON_PINS | PORTF_SWITCH_PINS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(GPIO_PORTF_AHB_BASE,
                         PORTF_BUTTON_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPU);
    MAP_GPIOPadConfigSet(GPIO_PORTF_AHB_BASE,
            PORTF_SWITCH_PINS,
                         GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPD);
}

//*****************************************************************************
//
// Initialize the ADC inputs used by the game pad device.
//
//*****************************************************************************
void
ADCInit(void)
{
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOB);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOD);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);
    SysCtlGPIOAHBEnable(SYSCTL_PERIPH_GPIOE);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_ADC0);
    SysCtlPeripheralReset(SYSCTL_PERIPH_ADC0);

    //
    // Select the external reference for greatest accuracy.
    //
    ADCReferenceSet(ADC0_BASE, ADC_REF_EXT_3V);

    //
    // Configure the pins to be used as analog inputs.
    //
    GPIOPinTypeADC(GPIO_PORTB_AHB_BASE, GPIO_PIN_5);
    GPIOPinTypeADC(GPIO_PORTD_AHB_BASE, GPIO_PIN_3 | GPIO_PIN_2 | GPIO_PIN_1 |
                   GPIO_PIN_0);
    GPIOPinTypeADC(GPIO_PORTE_AHB_BASE, GPIO_PIN_5 | GPIO_PIN_2 | GPIO_PIN_1);

    //
    // Configure the sequence step
    //
    ADCSequenceStepConfigure(ADC0_BASE, 0, 0, ADC_CTL_CH1);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 1, ADC_CTL_CH2);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 2, ADC_CTL_CH4);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 3, ADC_CTL_CH5);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 4, ADC_CTL_CH6);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 5, ADC_CTL_CH7);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 6, ADC_CTL_CH8);
    ADCSequenceStepConfigure(ADC0_BASE, 0, 7, ADC_CTL_CH11 | ADC_CTL_IE |
                                              ADC_CTL_END);

    ADCSequenceEnable(ADC0_BASE, 0);
}

#define Convert10Bit(ui32Value) ((int16_t)((0x7ff - ui32Value) >> 2))
#define Convert8Bit(ui32Value)  ((int8_t)((0x7ff - ui32Value) >> 4))

//*****************************************************************************
//
// This is the main loop that runs the application.
//
//*****************************************************************************
int
main(void)
{
    uint32_t ui32ButtonsChanged, ui32ButtonRaw;
    bool bUpdateA, bUpdateB;

    //
    // Set the clocking to run from the PLL at 50MHz
    //
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_4 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN |
                       SYSCTL_XTAL_16MHZ);

    //
    // Open UART0 and show the application name on the UART.
    //
    ConfigureUART();

    UARTprintf("\033[2JTiva C Series USB gamepad composite device example\n");
    UARTprintf("---------------------------------\n\n");

    //
    // Not configured initially.
    //
    g_iGamepadState = eStateNotConfigured;

    //
    // Enable the GPIO peripheral used for USB, and configure the USB
    // pins.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    ROM_GPIOPinTypeUSBAnalog(GPIO_PORTD_BASE, GPIO_PIN_4 | GPIO_PIN_5);

    //
    // Enable the system tick.
    //
    ROM_SysTickPeriodSet(ROM_SysCtlClockGet() / SYSTICKS_PER_SECOND);
    ROM_SysTickIntEnable();
    ROM_SysTickEnable();

    //
    // Configure the GPIOS for the buttons.
    //
    GamepadButtonsInit();

    //
    // Initialize the ADC channels.
    //
    ADCInit();

    //
    // Tell the user what we are up to.
    //
    UARTprintf("Configuring USB\n");

    //
    // Set the USB stack mode to Device mode.
    //
    USBStackModeSet(0, eUSBModeForceDevice, 0);

    //
    // Pass our device information to the USB HID device class driver,
    // initialize the USB
    // controller and connect the device to the bus.
    //
    USBDHIDGamepadCompositeInit(0, &g_sGamepadDeviceA, &g_psCompDevices[0]);
    USBDHIDGamepadCompositeInit(0, &g_sGamepadDeviceB, &g_psCompDevices[1]);

    //
    // Pass the device information to the USB library and place the device
    // on the bus.
    //
    USBDCompositeInit(0, &g_sCompGameDevice, DESCRIPTOR_DATA_SIZE,
                      g_pui8DescriptorData);


    //
    // Initialize the reports to 0.
    //
    sReportA.ui16Buttons = 0;
    sReportA.i16XPos = 0;
    sReportA.i16YPos = 0;
    sReportA.i16RXPos = 0;
    sReportA.i16YPos = 0;

    sReportB.ui16Buttons = 0;
    sReportB.i16XPos = 0;
    sReportB.i16YPos = 0;
    sReportB.i16RXPos = 0;
    sReportB.i16YPos = 0;

    ADCProcessorTrigger(ADC0_BASE, 0);

    //
    // The main loop starts here.  We begin by waiting for a host connection
    // then drop into the main keyboard handling section.  If the host
    // disconnects, we return to the top and wait for a new connection.
    //
    while(1)
    {
        //
        // Wait here until USB device is connected to a host.
        //
        if(g_iGamepadState == eStateIdle)
        {
            //
            // No update by default.
            //
            bUpdateA = false;
            bUpdateB = false;

            //
            // See if the buttons updated.
            //
            GamepadButtonsGet(&ui32ButtonsChanged, &ui32ButtonRaw);

            //
            // See if buttons on Gamepad A changed.
            //
            if(ui32ButtonsChanged & BUTTON_MASK_A)
            {
                bUpdateA = true;
                sReportA.ui16Buttons = ui32ButtonRaw & BUTTON_MASK_A;
            }

            //
            // See if buttons on Gamepad B changed.
            //
            if(ui32ButtonsChanged & BUTTON_MASK_B)
            {
                bUpdateB = true;
                sReportB.ui16Buttons = (ui32ButtonRaw & BUTTON_MASK_B) >> 16;
            }

            //
            // See if the ADC updated.
            //
            if(ADCIntStatus(ADC0_BASE, 0, false) != 0)
            {
                //
                // Clear the ADC interrupt.
                //
                ADCIntClear(ADC0_BASE, 0);

                //
                // Read the data and trigger a new sample request.
                //
                ADCSequenceDataGet(ADC0_BASE, 0, &g_pui32ADCData[0]);
                ADCProcessorTrigger(ADC0_BASE, 0);

                //
                // Update the reports.
                //
                sReportA.i16XPos = Convert10Bit(g_pui32ADCData[0]);
                sReportA.i16YPos = Convert10Bit(g_pui32ADCData[1]);
                sReportA.i16RXPos = Convert10Bit(g_pui32ADCData[2]);
                sReportA.i16RYPos = Convert10Bit(g_pui32ADCData[3]);
                sReportB.i16XPos = Convert10Bit(g_pui32ADCData[4]);
                sReportB.i16YPos = Convert10Bit(g_pui32ADCData[5]);
                sReportB.i16RXPos = Convert10Bit(g_pui32ADCData[6]);
                sReportB.i16RYPos = Convert10Bit(g_pui32ADCData[7]);
                bUpdateA = true;
                bUpdateB = true;
            }

            //
            // The state below can change in interrupt context and must be
            // protected here.
            //
            IntMasterDisable();

            //
            // Send the gamepad A report if there was an update and not
            // already sending.
            //
            if((bUpdateA) && ((g_iGamepadState != eStateSendingA) &&
                              (g_iGamepadState != eStateSendingAB)))
            {
                USBDHIDGamepadSendReport(&g_sGamepadDeviceA, &sReportA,
                                         sizeof(sReportA));

                //
                // Now sending data for gamepad A.
                //
                if(g_iGamepadState == eStateSendingB)
                {
                    g_iGamepadState = eStateSendingAB;
                }
                else
                {
                    g_iGamepadState = eStateSendingA;
                }
            }

            //
            // Send the gamepad B report if there was an update and not
            // already sending.
            //
            if((bUpdateB) && ((g_iGamepadState != eStateSendingB) &&
                              (g_iGamepadState != eStateSendingAB)))
            {
                USBDHIDGamepadSendReport(&g_sGamepadDeviceB, &sReportB,
                                         sizeof(sReportB));
                //
                // Now sending data but protect this from an interrupt since
                // it can change in interrupt context as well.
                //
                if(g_iGamepadState == eStateSendingA)
                {
                    g_iGamepadState = eStateSendingAB;
                }
                else
                {
                    g_iGamepadState = eStateSendingB;
                }

            }

            //
            // Restore interrupts.
            //
            IntMasterEnable();
        }
    }
}
