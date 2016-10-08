//*****************************************************************************
//
// pressure_bmp180.c - Example to demonstrate use of the SensorLib with the
//                     BMP180
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
// This is part of revision 2.1.3.156 of the DK-TM4C129X Firmware Package.
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
#include "grlib/grlib.h"
#include "drivers/frame.h"
#include "drivers/kentec320x240x16_ssd2119.h"
#include "drivers/pinout.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "sensorlib/hw_bmp180.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/bmp180.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Pressure Measurement with the BMP180 (pressure_bmp180)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, DK-TM4C129X
//! and the SensHub BoosterPack to obtain air pressure and temperature
//! measurements with the BMP180 sensor.
//!
//! The raw sensor measurements are printed to LCD and terminal.  Connect a
//! serial terminal program to the DK-TM4C129X's ICDI virtual serial port at
//! 115,200 baud.  Use eight bits per byte, no parity and one stop bit.  The
//! blue LED blinks every time data is read from the BMP180 sensor.
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
    if(ui8Status == I2CM_STATUS_SUCCESS)
    {
        g_vui8DataFlag = 1;
    }
}

//*****************************************************************************
//
// Called by the NVIC as a result of I2C3 Interrupt. I2C3 is the I2C connection
// to the BMP180.
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
// sample interval
//
//*****************************************************************************
void
SysTickIntHandler()
{
    //
    // Blink the blue LED to indicate data read from BMP180.
    //
    MAP_GPIOPinWrite(GPIO_PORTQ_BASE, GPIO_PIN_4,
                     ((GPIOPinRead(GPIO_PORTQ_BASE, GPIO_PIN_4)) ^
                      GPIO_PIN_4));
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
    tContext sContext;
    uint32_t ui32SysClock;
    char pcBuf[15];

    //
    // Setup the system clock to run at 40 Mhz from PLL with crystal reference
    //
    ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                           SYSCTL_OSC_MAIN |
                                           SYSCTL_USE_PLL |
                                           SYSCTL_CFG_VCO_480), 40000000);

    //
    // Configure the device pins.
    //
    PinoutSet();

    //
    // Initialize the display driver.
    //
    Kentec320x240x16_SSD2119Init(ui32SysClock);

    //
    // Initialize the graphics context.
    //
    GrContextInit(&sContext, &g_sKentec320x240x16_SSD2119);

    //
    // Draw the application frame.
    //
    FrameDraw(&sContext, "bmp180");

    //
    // Flush any cached drawing operations.
    //
    GrFlush(&sContext);

    //
    // Enable UART0
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    //
    // Initialize the UART for console I/O.
    //
    UARTStdioConfig(0, 115200, ui32SysClock);

    //
    // Print the welcome message to the terminal.
    //
    UARTprintf("\033[2JBMP180 Example\n");

    //
    // The I2C3 peripheral must be enabled before use.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C3);

    //
    // Configure the pin muxing for I2C3 functions on port G4 and G5.
    // This step is not necessary if your part does not support pin muxing.
    //
    MAP_GPIOPinConfigure(GPIO_PG4_I2C3SCL);
    MAP_GPIOPinConfigure(GPIO_PG5_I2C3SDA);

    //
    // Select the I2C function for these pins.  This function will also
    // configure the GPIO pins pins for I2C operation, setting them to
    // open-drain operation with weak pull-ups.  Consult the data sheet
    // to see which functions are allocated per pin.
    //
    MAP_GPIOPinTypeI2CSCL(GPIO_PORTG_BASE, GPIO_PIN_4);
    MAP_GPIOPinTypeI2C(GPIO_PORTG_BASE, GPIO_PIN_5);

    //
    // Enable interrupts to the processor.
    //
    MAP_IntMasterEnable();

    //
    // Initialize I2C3 peripheral.
    //
    I2CMInit(&g_sI2CInst, I2C3_BASE, INT_I2C3, 0xff, 0xff,
             ui32SysClock);

    //
    // Initialize the BMP180
    //
    BMP180Init(&g_sBMP180Inst, &g_sI2CInst, BMP180_I2C_ADDRESS,
               BMP180AppCallback, &g_sBMP180Inst);

    //
    // Wait for initialization callback to indicate reset request is complete.
    //
    while(g_vui8DataFlag == 0)
    {
        //
        // Wait for I2C Transactions to complete.
        //
    }

    //
    // Reset the data ready flag
    //
    g_vui8DataFlag = 0;

    //
    // Enable the system ticks at 10 hz.
    //
    MAP_SysTickPeriodSet(ui32SysClock / (10 * 3));
    MAP_SysTickIntEnable();
    MAP_SysTickEnable();

    //
    // Configure PQ4 to control the blue LED.
    //
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTQ_BASE, GPIO_PIN_4);

    //
    // Print temperature, pressure and altitude labels once on the LCD.
    //
    GrStringDraw(&sContext, "Temperature", 11,
                 ((GrContextDpyWidthGet(&sContext) / 2) - 96),
                 ((GrContextDpyHeightGet(&sContext) - 32) / 2) - 24, 1);
    GrStringDraw(&sContext, "Pressure", 8,
                 ((GrContextDpyWidthGet(&sContext) / 2) - 63),
                 (GrContextDpyHeightGet(&sContext) - 32) / 2, 1);
    GrStringDraw(&sContext, "Altitude", 8,
                 ((GrContextDpyWidthGet(&sContext) / 2) - 59),
                 ((GrContextDpyHeightGet(&sContext) - 32) / 2) + 24, 1);

    //
    // Begin the data collection and printing.  Loop Forever.
    //
    while(1)
    {
        //
        // Read the data from the BMP180 over I2C.  This command starts a
        // temperature measurement.  Then polls until temperature is ready.
        // Then automatically starts a pressure measurement and polls for that
        // to complete.  When both measurement are complete and in the local
        // buffer then the application callback is called from the I2C
        // interrupt context.  Polling is done on I2C interrupts allowing
        // processor to continue doing other tasks as needed.
        //
        BMP180DataRead(&g_sBMP180Inst, BMP180AppCallback, &g_sBMP180Inst);
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
        // Get a local copy of the latest temperature data in float format.
        //
        BMP180DataTemperatureGetFloat(&g_sBMP180Inst, &fTemperature);

        //
        // Convert the floats to an integer part and fraction part for easy
        // print.
        //
        i32IntegerPart = (int32_t) fTemperature;
        i32FractionPart =(int32_t) (fTemperature * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print temperature with three digits of decimal precision to LCD and
        // terminal.
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%03d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&sContext, pcBuf, 8,
                     ((GrContextDpyWidthGet(&sContext) / 2) + 16),
                     ((GrContextDpyHeightGet(&sContext) - 32) / 2) - 24, 1);
        UARTprintf("Temperature %3d.%03d\t\t", i32IntegerPart,
                                               i32FractionPart);

        //
        // Get a local copy of the latest air pressure data in float format.
        //
        BMP180DataPressureGetFloat(&g_sBMP180Inst, &fPressure);

        //
        // Convert the floats to an integer part and fraction part for easy
        // print.
        //
        i32IntegerPart = (int32_t) fPressure;
        i32FractionPart =(int32_t) (fPressure * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print Pressure with three digits of decimal precision to LCD and
        // terminal.
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&sContext, pcBuf, -1,
                     ((GrContextDpyWidthGet(&sContext) / 2) + 16),
                     (GrContextDpyHeightGet(&sContext) - 32) / 2, 1);
        UARTprintf("Pressure %3d.%03d\t\t", i32IntegerPart, i32FractionPart);

        //
        // Calculate the altitude.
        //
        fAltitude = 44330.0f * (1.0f - powf(fPressure / 101325.0f,
                                            1.0f / 5.255f));

        //
        // Convert the floats to an integer part and fraction part for easy
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
        // Print altitude with three digits of decimal precision to LCD and
        // terminal.
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&sContext, pcBuf, 8,
                     ((GrContextDpyWidthGet(&sContext) / 2) + 16),
                     ((GrContextDpyHeightGet(&sContext) - 32) / 2) + 24, 1);
        UARTprintf("Altitude %3d.%03d", i32IntegerPart, i32FractionPart);

        //
        // Print new line.
        //
        UARTprintf("\n");

        //
        // Delay to keep printing speed reasonable. About 100 milliseconds.
        //
        MAP_SysCtlDelay(ui32SysClock / (10 * 3));
    }
}
