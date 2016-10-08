//*****************************************************************************
//
// humidity_sht21.c - Example to demonstrate minimal humidity measurement with
//                    SensorLib the SHT21 and SensHub BoosterPack
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
#include "grlib/grlib.h"
#include "drivers/frame.h"
#include "drivers/kentec320x240x16_ssd2119.h"
#include "drivers/pinout.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "sensorlib/hw_sht21.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/sht21.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Humidity Measurement with the SHT21 (humidity_sht21)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, DK-TM4C129X
//! and SensHub BoosterPack to obtain temperature and relative humidity of the
//! environment using the Sensirion SHT21 sensor.
//!
//! The humidity and temperature as measured by the SHT21 is printed to LCD and
//! terminal. Connect a serial terminal program to the DK-TM4C129X's ICDI
//! virtual serial port at 115,200 baud.  Use eight bits per byte, no parity
//! and one stop bit.  The blue LED blinks to indicate the code is running.
//
//*****************************************************************************


//*****************************************************************************
//
// Define SHT21 I2C Address.
//
//*****************************************************************************
#define SHT21_I2C_ADDRESS  0x40

//*****************************************************************************
//
// Structure to hold the graphics context.
//
//*****************************************************************************
tContext g_sContext;

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Global instance structure for the SHT21 sensor driver.
//
//*****************************************************************************
tSHT21 g_sSHT21Inst;

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
// SHT21 Sensor callback function.  Called at the end of SHT21 sensor driver
// transactions. This is called from I2C interrupt context. Therefore, we just
// set a flag and let main do the bulk of the computations and display.
//
//*****************************************************************************
void
SHT21AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
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
SHT21AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
{
    char pcBuf[50];

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
    // Print the same error message to the LCD in Red.
    //
    usnprintf(pcBuf, sizeof(pcBuf), "Error: %d, File: %s, Line: %d",
              g_vui8ErrorFlag, pcFilename, ui32Line);
    GrContextForegroundSet(&g_sContext, ClrRed);
    GrStringDrawCentered(&g_sContext, pcBuf, -1,
                         GrContextDpyWidthGet(&g_sContext) / 2,
                         ((GrContextDpyHeightGet(&g_sContext) - 32) - 24),
                         1);

    //
    // Go to sleep wait for interventions.  A more robust application could
    // attempt corrective actions here.
    //
    while(1)
    {
        MAP_SysCtlSleep();
    }
}

//*****************************************************************************
//
// Called by the NVIC as a result of I2C3 Interrupt. I2C3 is the I2C connection
// to the SHT21.
//
//*****************************************************************************
void
SHT21I2CIntHandler(void)
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
// Function to wait for the SHT21 transactions to complete.
//
//*****************************************************************************
void
SHT21AppI2CWait(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Put the processor to sleep while we wait for the I2C driver to
    // indicate that the transaction is complete.
    //
    while((g_vui8DataFlag == 0) && (g_vui8ErrorFlag == 0))
    {
        MAP_SysCtlSleep();
    }

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ErrorFlag)
    {
        SHT21AppErrorHandler(pcFilename, ui32Line);
    }

    //
    // clear the data flag for next use.
    //
    g_vui8DataFlag = 0;
}

//*****************************************************************************
//
// Main 'C' Language entry point.
//
//*****************************************************************************
int
main(void)
{
    float fTemperature, fHumidity;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;
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
    GrContextInit(&g_sContext, &g_sKentec320x240x16_SSD2119);

    //
    // Draw the application frame.
    //
    FrameDraw(&g_sContext, "sht21");

    //
    // Flush any cached drawing operations.
    //
    GrFlush(&g_sContext);

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
    UARTprintf("\033[2JSHT21 Example\n");

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
    // Keep only some parts of the systems running while in sleep mode.
    // UART0 is the virtual serial port.
    // I2C3 is the I2C interface to the TMP006.
    //
    MAP_SysCtlPeripheralClockGating(true);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_UART0);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_I2C3);

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
    // Initialize the TMP006
    //
    SHT21Init(&g_sSHT21Inst, &g_sI2CInst, SHT21_I2C_ADDRESS,
              SHT21AppCallback, &g_sSHT21Inst);

    //
    // Wait for the I2C transactions to complete before moving forward
    //
    SHT21AppI2CWait(__FILE__, __LINE__);

    //
    // Delay for 20 milliseconds for SHT21 reset to complete itself.
    // Datasheet says reset can take as long 15 milliseconds.
    //
    MAP_SysCtlDelay(ui32SysClock / (50 * 3));

    //
    // Configure PQ4 to control the blue LED.
    //
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTQ_BASE, GPIO_PIN_4);

    //
    // Print humidity and temperature labels once on the LCD.
    //
    GrStringDraw(&g_sContext, "Humidity", 8,
                 ((GrContextDpyWidthGet(&g_sContext) / 2) - 59),
                 (GrContextDpyHeightGet(&g_sContext) - 32) / 2, 1);
    GrStringDraw(&g_sContext, "Temperature", 11,
                 ((GrContextDpyWidthGet(&g_sContext) / 2) - 91),
                 ((GrContextDpyHeightGet(&g_sContext) - 32) / 2) + 24, 1);

    //
    // Loop Forever
    //
    while(1)
    {
        //
        // Blink the blue LED to indicate activity.
        //
        MAP_GPIOPinWrite(GPIO_PORTQ_BASE, GPIO_PIN_4,
                         ((GPIOPinRead(GPIO_PORTQ_BASE, GPIO_PIN_4)) ^
                          GPIO_PIN_4));

        //
        // Write the command to start a humidity measurement
        //
        SHT21Write(&g_sSHT21Inst, SHT21_CMD_MEAS_RH, g_sSHT21Inst.pui8Data, 0,
                   SHT21AppCallback, &g_sSHT21Inst);

        //
        // Wait for the I2C transactions to complete before moving forward
        //
        SHT21AppI2CWait(__FILE__, __LINE__);

        //
        // Wait 33 milliseconds before attempting to get the result. Datasheet
        // claims this can take as long as 29 milliseconds
        //
        MAP_SysCtlDelay(ui32SysClock / (30 * 3));

        //
        // Get the raw data from the sensor over the I2C bus
        //
        SHT21DataRead(&g_sSHT21Inst, SHT21AppCallback, &g_sSHT21Inst);

        //
        // Wait for the I2C transactions to complete before moving forward
        //
        SHT21AppI2CWait(__FILE__, __LINE__);

        //
        // Get a copy of the most recent raw data in floating point format.
        //
        SHT21DataHumidityGetFloat(&g_sSHT21Inst, &fHumidity);

        //
        // Write the command to start a temperature measurement
        //
        SHT21Write(&g_sSHT21Inst, SHT21_CMD_MEAS_T, g_sSHT21Inst.pui8Data, 0,
                   SHT21AppCallback, &g_sSHT21Inst);

        //
        // Wait for the I2C transactions to complete before moving forward
        //
        SHT21AppI2CWait(__FILE__, __LINE__);

        //
        // Wait 100 milliseconds before attempting to get the result. Datasheet
        // claims this can take as long as 85 milliseconds
        //
        MAP_SysCtlDelay(ui32SysClock / (10 * 3));

        //
        // Read the conversion data from the sensor over I2C.
        //
        SHT21DataRead(&g_sSHT21Inst, SHT21AppCallback, &g_sSHT21Inst);

        //
        // Wait for the I2C transactions to complete before moving forward
        //
        SHT21AppI2CWait(__FILE__, __LINE__);

        //
        // Get the most recent temperature result as a float in celcius.
        //
        SHT21DataTemperatureGetFloat(&g_sSHT21Inst, &fTemperature);

        //
        // Convert the floats to an integer part and fraction part for easy
        // print. Humidity is returned as 0.0 to 1.0 so multiply by 100 to get
        // percent humidity.
        //
        fHumidity *= 100.0f;
        i32IntegerPart = (int32_t) fHumidity;
        i32FractionPart = (int32_t) (fHumidity * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print the humidity value using the integers we just created
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&g_sContext, pcBuf, 8,
                     ((GrContextDpyWidthGet(&g_sContext) / 2) + 16),
                     (GrContextDpyHeightGet(&g_sContext) - 32) / 2, 1);
        UARTprintf("Humidity %3d.%03d\t", i32IntegerPart, i32FractionPart);

        //
        // Perform the conversion from float to a printable set of integers
        //
        i32IntegerPart = (int32_t) fTemperature;
        i32FractionPart = (int32_t) (fTemperature * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print the temperature as integer and fraction parts.
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&g_sContext, pcBuf, 8,
                     ((GrContextDpyWidthGet(&g_sContext) / 2) + 16),
                     ((GrContextDpyHeightGet(&g_sContext) - 32) / 2) + 24, 1);
        UARTprintf("Temperature %3d.%03d\n", i32IntegerPart, i32FractionPart);

        //
        // Delay for one second. This is to keep sensor duty cycle
        // to about 10% as suggested in the datasheet, section 2.4.
        // This minimizes self heating effects and keeps reading more accurate.
        //
        MAP_SysCtlDelay(ui32SysClock / 3);
    }
}
