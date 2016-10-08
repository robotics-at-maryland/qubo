//*****************************************************************************
//
// pressure_bmp180.c - Example to use of the SensorLib with the BMP180.
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
#include <math.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "sensorlib/hw_bmp180.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/bmp180.h"
#include "drivers/pinout.h"
#include "drivers/buttons.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Pressure Measurement with the BMP180 (pressure_bmp180)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, the
//! EK-TM4C1294XL LaunchPad, and the SensHub BoosterPack to obtain air pressure
//! and temperature measurements with the BMP180 sensor.
//!
//! SensHub BoosterPack (BOOSTXL-SENSHUB) must be installed on BoosterPack 1
//! interface headers.
//!
//! Instructions for use of SensorHub on BoosterPack 2 headers are in the code
//! comments.
//!
//! Connect a serial terminal program to the LaunchPad's ICDI virtual serial
//! port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
//! The raw sensor measurements are printed to the terminal.  The LED
//! blinks at 1 Hz once the initialization is complete and the example is
//! running.
//
//*****************************************************************************

//*****************************************************************************
//
// Define BMP180 I2C Address.
//
//*****************************************************************************
#define BMP180_I2C_ADDRESS      0x77

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Global instance structure for the BMP180 sensor driver.
//
//*****************************************************************************
tBMP180 g_sBMP180Inst;

//*****************************************************************************
//
// Global new data flag to alert main that BMP180 data is ready.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8DataFlag;

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
// BMP180 Sensor callback function.  Called at the end of BMP180 sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations and display.
//
//*****************************************************************************
void BMP180AppCallback(void* pvCallbackData, uint_fast8_t ui8Status)
{
    //
    // If the transaction was successful then set the data ready flag.
    if(ui8Status == I2CM_STATUS_SUCCESS)
    {
        g_vui8DataFlag = 1;
    }

    //
    // Turn off the LED to show read is complete.
    //
    LEDWrite(CLP_D1 | CLP_D2, 0);
}

//*****************************************************************************
//
// Called by the NVIC as a result of I2C7 Interrupt. I2C7 is the I2C connection
// to the BMP180.
//
// This handler is installed in the vector table for I2C7 by default.  To use
// the SensHub on BoosterPack 2 interface change the startup file to place this
// interrupt in I2C8 vector location.
//
//*****************************************************************************
void
BMP180I2CIntHandler(void)
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
// Called by the NVIC as a SysTick interrupt, which is used to generate the
// sample interval.
//
//*****************************************************************************
void
SysTickIntHandler()
{
    //
    // Turn on the LED to indicate we are reading.
    //
    LEDWrite(CLP_D1 | CLP_D2, CLP_D2);

    //
    // Start a read of data from the pressure sensor. BMP180AppCallback is
    // called when the read is complete.
    //
    BMP180DataRead(&g_sBMP180Inst, BMP180AppCallback, &g_sBMP180Inst);
}

//*****************************************************************************
//
// Main 'C' Language entry point.
//
//*****************************************************************************
int
main(void)
{
    float fTemperature, fPressure, fAltitude;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;
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
    // Clear the terminal and print the welcome message.
    //
    UARTprintf("\033[2J\033[H");
    UARTprintf("BMP180 Example\n");

    //
    // The I2C7 peripheral must be enabled before use.
    //
    // For BoosterPack 2 interface change this to I2C8.
    // Change the GPIO port to GPIO port A.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C7);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);

    //
    // Configure the pin muxing for I2C7 functions on port D0 and D1.
    // This step is not necessary if your part does not support pin muxing.
    //
    // For BoosterPack 2 interface change these to PA3_I2C8SDA and PA2_I2C8SCL.
    //
    ROM_GPIOPinConfigure(GPIO_PD0_I2C7SCL);
    ROM_GPIOPinConfigure(GPIO_PD1_I2C7SDA);

    //
    // Select the I2C function for these pins.  This function will also
    // configure the GPIO pins for I2C operation, setting them to
    // open-drain operation with weak pull-ups.  Consult the data sheet
    // to see which functions are allocated per pin.
    //
    // For BoosterPack 2 interface change these to PA2 (SCL) and PA3 (SDA).
    //
    GPIOPinTypeI2CSCL(GPIO_PORTD_BASE, GPIO_PIN_0);
    ROM_GPIOPinTypeI2C(GPIO_PORTD_BASE, GPIO_PIN_1);

    //
    // Enable interrupts to the processor.
    //
    ROM_IntMasterEnable();

    //
    // Initialize I2C7 peripheral.
    //
    // For BoosterPack 2 change these to I2C8.
    //
    I2CMInit(&g_sI2CInst, I2C7_BASE, INT_I2C7, 0xff, 0xff, ui32SysClock);

    //
    // Turn on the LED to indicate we are performing BMP180 init.
    // Turned off automatically inside the application callback for the BMP180
    // init is when complete.
    //
    LEDWrite(CLP_D1 | CLP_D2, CLP_D2);

    //
    // Initialize the BMP180.
    //
    BMP180Init(&g_sBMP180Inst, &g_sI2CInst, BMP180_I2C_ADDRESS,
               BMP180AppCallback, &g_sBMP180Inst);

    //
    // Wait for initialization callback to indicate reset request is complete.
    //
    while(g_vui8DataFlag == 0)
    {
        //
        // Wait for I2C transactions to complete.
        //
    }

    //
    // Reset the data ready flag.
    //
    g_vui8DataFlag = 0;

    //
    // Enable the system ticks at 10 Hz.
    //
    ROM_SysTickPeriodSet(ui32SysClock / (10 * 3));
    ROM_SysTickIntEnable();
    ROM_SysTickEnable();

    //
    // Begin the data collection and printing.  Loop Forever.
    //
    while(1)
    {
        //
        // The reads are started by SysTick Interrupt, we poll here to detect
        // when a read is complete.
        //
        while(g_vui8DataFlag == 0)
        {
            //
            // Wait for the new data set to be available.
            //
        }

        //
        // Reset the data ready flag.
        //
        g_vui8DataFlag = 0;

        //
        // Get a local copy of the latest temperature and pressure data in
        // float format.
        //
        BMP180DataTemperatureGetFloat(&g_sBMP180Inst, &fTemperature);
        BMP180DataPressureGetFloat(&g_sBMP180Inst, &fPressure);

        //
        // Convert the temperature to an integer part and fraction part for
        // easy print.
        //
        i32IntegerPart = (int32_t) fTemperature;
        i32FractionPart =(int32_t) (fTemperature * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print temperature with three digits of decimal precision.
        //
        UARTprintf("Temperature %3d.%03d\t\t", i32IntegerPart,
                   i32FractionPart);

        //
        // Convert the pressure to an integer part and fraction part for
        // easy print.
        //
        i32IntegerPart = (int32_t) fPressure;
        i32FractionPart =(int32_t) (fPressure * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print Pressure with three digits of decimal precision.
        //
        UARTprintf("Pressure %3d.%03d\t\t", i32IntegerPart, i32FractionPart);

        //
        // Calculate the altitude.
        //
        fAltitude = 44330.0f * (1.0f - powf(fPressure / 101325.0f,
                                            1.0f / 5.255f));

        //
        // Convert the altitude to an integer part and fraction part for easy
        // print.
        //
        i32IntegerPart = (int32_t) fAltitude;
        i32FractionPart =(int32_t) (fAltitude * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print altitude with three digits of decimal precision.
        //
        UARTprintf("Altitude %3d.%03d", i32IntegerPart, i32FractionPart);

        //
        // Print new line.
        //
        UARTprintf("\n");
    }//while end
}
