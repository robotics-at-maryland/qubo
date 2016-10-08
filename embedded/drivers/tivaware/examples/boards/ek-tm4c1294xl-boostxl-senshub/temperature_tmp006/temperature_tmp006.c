//*****************************************************************************
//
// temperature_tmp006.c - Example to use of the SensorLib with the TMP006
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

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_ints.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "sensorlib/hw_tmp006.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/tmp006.h"
#include "drivers/pinout.h"
#include "drivers/buttons.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Temperature Measurement with the TMP006 (temperature_tmp006)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, TM4C1294
//! Connected LaunchPad and the SensHub BoosterPack to obtain ambient and object
//! temperature measurements with the Texas Instruments TMP006 sensor.
//! 
//! SensHub BoosterPack (BOOSTXL-SENSHUB) Must be installed on BoosterPack 1
//! interface headers.
//!
//! Connect a serial terminal program to the LaunchPad's ICDI virtual serial
//! port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
//! The raw sensor measurements are printed to the terminal.  An LED blinks at
//! 1Hz once the initialization is complete and the example is running.
//
//*****************************************************************************

//*****************************************************************************
//
// Define TMP006 I2C Address.
//
//*****************************************************************************
#define TMP006_I2C_ADDRESS      0x41

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Global instance structure for the TMP006 sensor driver.
//
//*****************************************************************************
tTMP006 g_sTMP006Inst;

//*****************************************************************************
//
// Global new data flag to alert main that TMP006 data is ready.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8DataFlag;

//*****************************************************************************
//
// Global new error flag to store the error condition if encountered.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8ErrorFlag;

//*****************************************************************************
//
// Application function to capture ASSERT failures and other debug conditions.
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
// TMP006 Sensor callback function.  Called at the end of TMP006 sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations and display.
//
//*****************************************************************************
void
TMP006AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
{
    //
    // If the transaction succeeded set the data flag to indicate to
    // application that this transaction is complete and data may be ready.
    //
    if(ui8Status == I2CM_STATUS_SUCCESS)
    {
        g_vui8DataFlag = 1;
    }

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8ErrorFlag = ui8Status;
}

//*****************************************************************************
//
// TMP006 Application error handler.
//
//*****************************************************************************
void
TMP006AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1m");
    UARTprintf("Error: %d, File: %s, Line: %d\n"
               "See I2C status definitions in utils\\i2cm_drv.h\n",
               g_vui8ErrorFlag, pcFilename, ui32Line);

    //
    // Return terminal color to normal
    //
    UARTprintf("\033[0m");

	//
    // Go to sleep wait for interventions.  A more robust application could
    // attempt corrective actions here.
    //
    while(1)
    {
        ROM_SysCtlSleep();
    }
}

//*****************************************************************************
//
// Called by the NVIC as a result of I2C Interrupt. I2C7 is the I2C connection
// to the TMP006 for BoosterPack 1 interface.  I2C8 must be used for 
// BoosterPack 2 interface.  Must also move this function pointer in the
// startup file interrupt vector table for your tool chain if using BoosterPack
// 2 interface headers.
//
//*****************************************************************************
void
TMP006I2CIntHandler(void)
{
    //
    // Pass through to the I2CM interrupt handler provided by sensor library.
    // This is required to be at application level so that I2CMIntHandler can
    // receive the instance structure pointer as an argument.
    //
    I2CMIntHandler(&g_sI2CInst);
}

//*****************************************************************************
//
// Called by the NVIC as a result of GPIO port H interrupt event. For this
// application GPIO port H pin 2 is the interrupt line for the TMP006
//
// To use the sensor hub on BoosterPack 2 modify this function to accept and 
// handle interrupts on GPIO Port P pin 5.  Also move the reference to this
// function in the startup file to GPIO Port P Int handler position in the
// vector table.
//
//*****************************************************************************
void
IntHandlerGPIOPortH(void)
{
    uint32_t ui32Status;

    ui32Status = GPIOIntStatus(GPIO_PORTH_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    GPIOIntClear(GPIO_PORTH_BASE, ui32Status);

    if(ui32Status & GPIO_PIN_2)
    {
        //
        // This interrupt indicates a conversion is complete and ready to be
        // fetched.  So we start the process of getting the data.
        //
        TMP006DataRead(&g_sTMP006Inst, TMP006AppCallback, &g_sTMP006Inst);
    }
}

//*****************************************************************************
//
// Main 'C' Language entry point.
//
//*****************************************************************************
int
main(void)
{
    float fAmbient, fObject;
    int_fast32_t i32IntegerPart;
    int_fast32_t i32FractionPart;
    uint32_t ui32LEDState;
    uint32_t ui32SysClock;

    //
    // Configure the system frequency.
    //
    ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                           SYSCTL_OSC_MAIN | SYSCTL_USE_PLL |
                                           SYSCTL_CFG_VCO_480), 120000000);

    //
    // Configure the device pins for this board.
    // This application does not use Ethernet or USB.
    //
    PinoutSet(false, false);

    //
    // Enable UART0
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, ui32SysClock);

    //
    // Print the welcome message to the terminal.
    //
    UARTprintf("\033[2J\033[1;1HTMP006 Example\n");

    //
    // The I2C7 peripheral must be enabled before use.
    //
    // For BoosterPack 2 interface use I2C8.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C7);

    //
    // Configure the pin muxing for I2C7 functions on port D0 and D1.
    // This step is not necessary if your part does not support pin muxing.
    //
    // For BoosterPack 2 interface use PA2 and PA3.
    // 
    ROM_GPIOPinConfigure(GPIO_PD0_I2C7SCL);
    ROM_GPIOPinConfigure(GPIO_PD1_I2C7SDA);

    //
    // Select the I2C function for these pins.  This function will also
    // configure the GPIO pins pins for I2C operation, setting them to
    // open-drain operation with weak pull-ups.  Consult the data sheet
    // to see which functions are allocated per pin.
    //
    // For BoosterPack 2 interface use PA2 and PA3.
    // 
    GPIOPinTypeI2CSCL(GPIO_PORTD_BASE, GPIO_PIN_0);
    ROM_GPIOPinTypeI2C(GPIO_PORTD_BASE, GPIO_PIN_1);

    //
    // Configure and Enable the GPIO interrupt. Used for DRDY from the TMP006
    //
    // For BoosterPack 2 interface use PP5.
    //
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTH_BASE, GPIO_PIN_2);
    GPIOIntEnable(GPIO_PORTH_BASE, GPIO_PIN_2);
    ROM_GPIOIntTypeSet(GPIO_PORTH_BASE, GPIO_PIN_2, GPIO_FALLING_EDGE);
    ROM_IntEnable(INT_GPIOH);

    //
    // Keep only some parts of the systems running while in sleep mode.
    // GPIOE is for the TMP006 data ready interrupt.
    // UART0 is the virtual serial port
    // TIMER0, TIMER1 and WTIMER5 are used by the RGB driver
    // I2C7 is the I2C interface to the TMP006
    //
    // For BoosterPack 2 Interface use GPIOP and I2C8
    //
    ROM_SysCtlPeripheralClockGating(true);
    ROM_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_GPIOH);
    ROM_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_UART0);
    ROM_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_I2C7);

    //
    // Enable interrupts to the processor.
    //
    ROM_IntMasterEnable();

    //
    // Initialize I2C peripheral.
    //
    // For BoosterPack 2 interface use I2C8
    //
    I2CMInit(&g_sI2CInst, I2C7_BASE, INT_I2C7, 0xff, 0xff, ui32SysClock);

    //
    // Initialize the TMP006
    //
    TMP006Init(&g_sTMP006Inst, &g_sI2CInst, TMP006_I2C_ADDRESS,
               TMP006AppCallback, &g_sTMP006Inst);

    //
    // Put the processor to sleep while we wait for the I2C driver to
    // indicate that the transaction is complete.
    //
    while((g_vui8DataFlag == 0) && (g_vui8ErrorFlag == 0))
    {
        ROM_SysCtlSleep();
    }

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ErrorFlag)
    {
        TMP006AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // clear the data flag for next use.
    //
    g_vui8DataFlag = 0;

    //
    // Delay for 10 milliseconds for TMP006 reset to complete.
    // Not explicitly required. Datasheet does not say how long a reset takes.
    //
    ROM_SysCtlDelay(ui32SysClock / (100 * 3));

    //
    // Enable the DRDY pin indication that a conversion is in progress.
    //
    TMP006ReadModifyWrite(&g_sTMP006Inst, TMP006_O_CONFIG,
                          ~TMP006_CONFIG_EN_DRDY_PIN_M,
                          TMP006_CONFIG_EN_DRDY_PIN, TMP006AppCallback,
                          &g_sTMP006Inst);

    //
    // Wait for the DRDY enable I2C transaction to complete.
    //
    while((g_vui8DataFlag == 0) && (g_vui8ErrorFlag == 0))
    {
        ROM_SysCtlSleep();
    }

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ErrorFlag)
    {
        TMP006AppErrorHandler(__FILE__, __LINE__);
    }

    //
    // clear the data flag for next use.
    //
    g_vui8DataFlag = 0;

    //
    // Loop Forever
    //
    while(1)
    {
        //
        // Toggle LED 3 on each time through the loop.
        //
        LEDRead(&ui32LEDState);
        LEDWrite(CLP_D3, (ui32LEDState ^ CLP_D3));
        
        //
        // Put the processor to sleep while we wait for the TMP006 to
        // signal that data is ready.  Also continue to sleep while I2C
        // transactions get the raw data from the TMP006
        //
        while((g_vui8DataFlag == 0) && (g_vui8ErrorFlag == 0))
        {            
            ROM_SysCtlSleep();
        }
        
        //
        // If an error occurred call the error handler immediately.
        //
        if(g_vui8ErrorFlag)
        {
            TMP006AppErrorHandler(__FILE__, __LINE__);
        }

        //
        // Reset the flag
        //
        g_vui8DataFlag = 0;

        //
        // Get a local copy of the latest data in float format.
        //
        TMP006DataTemperatureGetFloat(&g_sTMP006Inst, &fAmbient, &fObject);

        //
        // Convert the floating point ambient temperature  to an integer part
        // and fraction part for easy printing.
        //
        i32IntegerPart = (int32_t)fAmbient;
        i32FractionPart = (int32_t)(fAmbient * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }
        UARTprintf("Ambient %3d.%03d\t", i32IntegerPart, i32FractionPart);

        //
        // Convert the floating point ambient temperature  to an integer part
        // and fraction part for easy printing.
        //
        i32IntegerPart = (int32_t)fObject;
        i32FractionPart = (int32_t)(fObject * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }
        UARTprintf("Object %3d.%03d\n", i32IntegerPart, i32FractionPart);
    }
}
