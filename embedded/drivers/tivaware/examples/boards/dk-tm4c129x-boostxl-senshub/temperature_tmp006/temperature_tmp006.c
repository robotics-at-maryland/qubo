//*****************************************************************************
//
// temperature_tmp006.c - Example to demonstrate use of the SensorLib with the
//                        TMP006.
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
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
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
#include "sensorlib/hw_tmp006.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/tmp006.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Temperature Measurement with the TMP006 (temperature_tmp006)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, DK-TM4C129X
//! and the SensHub BoosterPack to obtain ambient and object temperature
//! measurements with the Texas Instruments TMP006 sensor.
//!
//! Note that the jumper on J36 for PQ7 must be disconnect to GREEN led as PQ7
//! is also used as DRDY signal by TMP006.
//!
//! The raw sensor measurements are printed to LCD and terminal.  Connect a
//! serial terminal program to the DK-TM4C129X's ICDI virtual serial port at
//! 115,200 baud.  Use eight bits per byte, no parity and one stop bit.  The
//! blue LED blinks to indicate the code is running.
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
// Structure to hold the graphics context.
//
//*****************************************************************************
tContext g_sContext;

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
// to the TMP006.
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
// Called by the NVIC as a result of GPIOQ7 interrupt event. For this
// application GPIO port Q pin 7 is the interrupt line for the TMP006.
//
//*****************************************************************************
void
GPIOQ7IntHandler(void)
{
    uint32_t ui32Status;

    ui32Status = MAP_GPIOIntStatus(GPIO_PORTQ_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    MAP_GPIOIntClear(GPIO_PORTQ_BASE, ui32Status);

    if(ui32Status & GPIO_PIN_7)
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
    FrameDraw(&g_sContext, "tmp006");

    //
    // Flush any cached drawing operations.
    //
    GrFlush(&g_sContext);

    //
    // Reset and enable Port Q to override any configuration that PinoutSet()
    // would have performed on PQ7 as it is muxed between Green LED on the
    // DK-TM4C129X and a pin on sensor hub that is used by this example.
    //
    MAP_SysCtlPeripheralReset(SYSCTL_PERIPH_GPIOQ);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOQ);

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
    UARTprintf("\033[2J\033[1;1HTMP006 Example\n");

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
    // configure the GPIO pins for I2C operation, setting them to open-drain
    // operation with weak pull-ups.  Consult the data sheet to see which
    // functions are allocated per pin.
    //
    MAP_GPIOPinTypeI2CSCL(GPIO_PORTG_BASE, GPIO_PIN_4);
    MAP_GPIOPinTypeI2C(GPIO_PORTG_BASE, GPIO_PIN_5);

    //
    // Configure and Enable the GPIO interrupt. Used for DRDY from the TMP006
    //
    MAP_GPIOPinTypeGPIOInput(GPIO_PORTQ_BASE, GPIO_PIN_7);
    MAP_GPIOIntEnable(GPIO_PORTQ_BASE, GPIO_PIN_7);
    MAP_GPIOIntTypeSet(GPIO_PORTQ_BASE, GPIO_PIN_7, GPIO_FALLING_EDGE);
    IntEnable(INT_GPIOQ7);

    //
    // Keep only some parts of the systems running while in sleep mode.
    // GPIOQ is for the TMP006 data ready interrupt.
    // UART0 is the virtual serial port.
    // I2C3 is the I2C interface to the TMP006.
    //
    MAP_SysCtlPeripheralClockGating(true);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_GPIOQ);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_UART0);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_I2C3);

    //
    // Enable interrupts to the processor.
    //
    MAP_IntMasterEnable();

    //
    // Initialize I2C3 peripheral.
    //
    I2CMInit(&g_sI2CInst, I2C3_BASE, INT_I2C3, 0xff, 0xff, ui32SysClock);

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
        MAP_SysCtlSleep();
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
    MAP_SysCtlDelay(ui32SysClock / (100 * 3));

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
        MAP_SysCtlSleep();
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
    // Configure PQ4 to control the blue LED.
    //
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTQ_BASE, GPIO_PIN_4);

    //
    // Print ambient and object temperature labels once on the LCD.
    //
    GrStringDraw(&g_sContext, "Ambient", 7,
                 ((GrContextDpyWidthGet(&g_sContext) / 2) - 56),
                 (GrContextDpyHeightGet(&g_sContext) - 32) / 2, 1);
    GrStringDraw(&g_sContext, "Object", 6,
                 ((GrContextDpyWidthGet(&g_sContext) / 2) - 48),
                 ((GrContextDpyHeightGet(&g_sContext) - 32) / 2) + 24, 1);

    //
    // Loop Forever
    //
    while(1)
    {
        //
        // Put the processor to sleep while we wait for the TMP006 to
        // signal that data is ready.  Also continue to sleep while I2C
        // transactions get the raw data from the TMP006
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
            TMP006AppErrorHandler(__FILE__, __LINE__);
        }

        //
        // Reset the flag
        //
        g_vui8DataFlag = 0;

        //
        // Blink the blue LED to indicate activity.
        //
        MAP_GPIOPinWrite(GPIO_PORTQ_BASE, GPIO_PIN_4,
                         ((GPIOPinRead(GPIO_PORTQ_BASE, GPIO_PIN_4)) ^
                          GPIO_PIN_4));

        //
        // Get a local copy of the latest data in float format.
        //
        TMP006DataTemperatureGetFloat(&g_sTMP006Inst, &fAmbient, &fObject);

        //
        // Convert the floating point ambient temperature to an integer part
        // and fraction part for easy printing.
        //
        i32IntegerPart = (int32_t)fAmbient;
        i32FractionPart = (int32_t)(fAmbient * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print the ambient temperature reading to LCD and terminal.
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&g_sContext, pcBuf, 8,
                     ((GrContextDpyWidthGet(&g_sContext) / 2) + 16),
                     (GrContextDpyHeightGet(&g_sContext) - 32) / 2, 1);
        UARTprintf("Ambient %3d.%03d\t", i32IntegerPart, i32FractionPart);

        //
        // Convert the floating point object  temperature to an integer part
        // and fraction part for easy printing.
        //
        i32IntegerPart = (int32_t)fObject;
        i32FractionPart = (int32_t)(fObject * 1000.0f);
        i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
        if(i32FractionPart < 0)
        {
            i32FractionPart *= -1;
        }

        //
        // Print the object temperature reading to LCD and terminal.
        //
        usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d ", i32IntegerPart,
                                                     i32FractionPart);
        GrStringDraw(&g_sContext, pcBuf, 8,
                     ((GrContextDpyWidthGet(&g_sContext) / 2) + 16),
                     ((GrContextDpyHeightGet(&g_sContext) - 32) / 2) + 24, 1);
        UARTprintf("Object %3d.%03d\n", i32IntegerPart, i32FractionPart);
    }
}
