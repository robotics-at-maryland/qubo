//*****************************************************************************
//
// light_isl29023.c - Example to demonstrate use of the SensorLib with the
//                    ISL29023
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
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "grlib/grlib.h"
#include "drivers/frame.h"
#include "drivers/kentec320x240x16_ssd2119.h"
#include "drivers/pinout.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "sensorlib/hw_isl29023.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/isl29023.h"

//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Light Measurement with the ISL29023 (light_isl29023)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, DK-TM4C129X
//! and the SensHub BoosterPack to obtain ambient and infrared light
//! measurements with the ISL29023 sensor.
//!
//! Note that the jumper on J36 for PQ7 must be disconnect to GREEN led as PQ7
//! is also used as INT signal by ISL29023.
//!
//! The raw sensor measurements are printed to LCD and terminal. Connect a
//! serial terminal program to the DK-TM4C129X's ICDI virtual serial port at
//! 115,200 baud.  Use eight bits per byte, no parity and one stop bit.  The
//! blue LED blinks to indicate the code is running.
//
//*****************************************************************************

//*****************************************************************************
//
// Define ISL29023 I2C Address.
//
//*****************************************************************************
#define ISL29023_I2C_ADDRESS    0x44

//*****************************************************************************
//
// The system tick rate expressed both as ticks per second and a millisecond
// period.
//
//*****************************************************************************
#define SYSTICKS_PER_SECOND     1
#define SYSTICK_PERIOD_MS       (1000 / SYSTICKS_PER_SECOND)

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
// Global instance structure for the ISL29023 sensor driver.
//
//*****************************************************************************
tISL29023 g_sISL29023Inst;

//*****************************************************************************
//
// Global flags to alert main that ISL29023 data is ready or an error
// has occurred.
//
//*****************************************************************************
volatile unsigned long g_vui8DataFlag;
volatile unsigned long g_vui8ErrorFlag;
volatile unsigned long g_vui8IntensityFlag;

//*****************************************************************************
//
// Constants to hold the floating point version of the thresholds for each
// range setting. Numbers represent an 81% and 19 % threshold levels. This
// creates a +/- 1% hysteresis band between range adjustments.
//
//*****************************************************************************
const float g_fThresholdHigh[4] =
{
    810.0f, 3240.0f, 12960.0f, 64000.0f
};
const float g_fThresholdLow[4] =
{
    0.0f, 760.0f, 3040.0f, 12160.0f
};

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
// ISL29023 Sensor callback function.  Called at the end of ISL29023 sensor
// driver transactions. This is called from I2C interrupt context. Therefore,
// we just set a flag and let main do the bulk of the computations and display.
//
//*****************************************************************************
void
ISL29023AppCallback(void *pvCallbackData, uint_fast8_t ui8Status)
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
// Called by the NVIC as a result of I2C3 Interrupt. I2C3 is the I2C connection
// to the ISL29023.
//
//*****************************************************************************
void
ISL29023I2CIntHandler(void)
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
// Called by the NVIC as a result of GPIO port F interrupt event. For this
// application GPIO port F pin 3 is the interrupt line for the ISL29023
//
// For this application this is a very low priority interrupt, we want to
// get notification of light values outside our thresholds but it is not the
// most important thing.
//
//*****************************************************************************
void
GPIOFIntHandler(void)
{
    uint32_t ui32Status;

    ui32Status = MAP_GPIOIntStatus(GPIO_PORTF_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    MAP_GPIOIntClear(GPIO_PORTF_BASE, ui32Status);

    if(ui32Status & GPIO_PIN_3)
    {
        //
        // ISL29023 has indicated that the light level has crossed outside of
        // the intensity threshold levels set in INT_LT and INT_HT registers.
        //
        g_vui8IntensityFlag = 1;
    }
}

//*****************************************************************************
//
// Interrupt handler for the system tick counter.
//
//*****************************************************************************
void
SysTickIntHandler(void)
{
    //
    // Go get the latest data from the sensor.
    //
    ISL29023DataRead(&g_sISL29023Inst, ISL29023AppCallback, &g_sISL29023Inst);
}

//*****************************************************************************
//
// ISL29023 Application error handler.
//
//*****************************************************************************
void
ISL29023AppErrorHandler(char *pcFilename, uint_fast32_t ui32Line)
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
        //
        // Do Nothing
        //
    }
}

//*****************************************************************************
//
// Function to wait for the ISL29023 transactions to complete.
//
//*****************************************************************************
void
ISL29023AppI2CWait(char *pcFilename, uint_fast32_t ui32Line)
{
    //
    // Put the processor to sleep while we wait for the I2C driver to
    // indicate that the transaction is complete.
    //
    while((g_vui8DataFlag == 0) && (g_vui8ErrorFlag == 0))
    {
        //
        // Do Nothing
        //
    }

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ErrorFlag)
    {
        ISL29023AppErrorHandler(pcFilename, ui32Line);
    }

    //
    // clear the data flag for next use.
    //
    g_vui8DataFlag = 0;
}

//*****************************************************************************
//
// Intensity and Range Tracking Function.  This adjusts the range and interrupt
// thresholds as needed.  Uses an 80/20 rule. If light is greather then 80% of
// maximum value in this range then go to next range up. If less than 20% of
// potential value in this range go the next range down.
//
//*****************************************************************************
void
ISL29023AppAdjustRange(tISL29023 *pInst)
{
    float fAmbient;
    uint8_t ui8NewRange;

    ui8NewRange = g_sISL29023Inst.ui8Range;

    //
    // Get a local floating point copy of the latest light data
    //
    ISL29023DataLightVisibleGetFloat(&g_sISL29023Inst, &fAmbient);

    //
    // Check if we crossed the upper threshold.
    //
    if(fAmbient > g_fThresholdHigh[g_sISL29023Inst.ui8Range])
    {
        //
        // The current intensity is over our threshold so adjsut the range
        // accordingly
        //
        if(g_sISL29023Inst.ui8Range < ISL29023_CMD_II_RANGE_64K)
        {
            ui8NewRange = g_sISL29023Inst.ui8Range + 1;
        }
    }

    //
    // Check if we crossed the lower threshold
    //
    if(fAmbient < g_fThresholdLow[g_sISL29023Inst.ui8Range])
    {
        //
        // If possible go to the next lower range setting and reconfig the
        // thresholds.
        //
        if(g_sISL29023Inst.ui8Range > ISL29023_CMD_II_RANGE_1K)
        {
            ui8NewRange = g_sISL29023Inst.ui8Range - 1;
        }
    }

    //
    // If the desired range value changed then send the new range to the sensor
    //
    if(ui8NewRange != g_sISL29023Inst.ui8Range)
    {
        ISL29023ReadModifyWrite(&g_sISL29023Inst, ISL29023_O_CMD_II,
                                ~ISL29023_CMD_II_RANGE_M, ui8NewRange,
                                ISL29023AppCallback, &g_sISL29023Inst);

        //
        // Wait for transaction to complete
        //
        ISL29023AppI2CWait(__FILE__, __LINE__);
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
    float fAmbient;
    int32_t i32IntegerPart, i32FractionPart;
    uint8_t ui8Mask;
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
    FrameDraw(&g_sContext, "isl29023");

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
    UARTprintf("\033[2JISL29023 Example\n");

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
    // Configure and Enable the GPIO interrupt.  Used for INT signal from the
    // ISL29023
    //
    MAP_GPIOPinTypeGPIOInput(GPIO_PORTF_BASE, GPIO_PIN_3);
    MAP_GPIOIntEnable(GPIO_PORTF_BASE, GPIO_PIN_3);
    MAP_GPIOIntTypeSet(GPIO_PORTF_BASE, GPIO_PIN_3, GPIO_FALLING_EDGE);
    MAP_IntEnable(INT_GPIOF);

    //
    // Keep only some parts of the systems running while in sleep mode.
    // GPIOE is for the ISL29023 interrupt pin.
    // UART0 is the virtual serial port.
    // I2C3 is the I2C interface to the ISL29023.
    //
    MAP_SysCtlPeripheralClockGating(true);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_GPIOF);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_UART0);
    MAP_SysCtlPeripheralSleepEnable(SYSCTL_PERIPH_I2C3);

    //
    // Configure desired interrupt priorities.  Setting the I2C interrupt to be
    // of more priority than SysTick and the GPIO interrupt means those
    // interrupt routines can use the I2CM_DRV Application context does not use
    // I2CM_DRV API and GPIO and SysTick are at the same priority level. This
    // prevents re-entrancy problems with I2CM_DRV but keeps the MCU in sleep
    // state as much as possible. UART is at least priority so it can operate
    // in the background.
    //
    MAP_IntPrioritySet(INT_I2C3, 0x00);
    MAP_IntPrioritySet(FAULT_SYSTICK, 0x40);
    MAP_IntPrioritySet(INT_GPIOF, 0x80);
    MAP_IntPrioritySet(INT_UART0, 0x80);

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
    // Initialize the ISL29023 Driver.
    //
    ISL29023Init(&g_sISL29023Inst, &g_sI2CInst, ISL29023_I2C_ADDRESS,
                 ISL29023AppCallback, &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    ISL29023AppI2CWait(__FILE__, __LINE__);

    //
    // Configure the ISL29023 to measure ambient light continuously. Set a 8
    // sample persistence before the INT pin is asserted. Clears the INT flag.
    // Persistence setting of 8 is sufficient to ignore camera flashes.
    //
    ui8Mask = (ISL29023_CMD_I_OP_MODE_M | ISL29023_CMD_I_INT_PERSIST_M |
               ISL29023_CMD_I_INT_FLAG_M);
    ISL29023ReadModifyWrite(&g_sISL29023Inst, ISL29023_O_CMD_I, ~ui8Mask,
                            (ISL29023_CMD_I_OP_MODE_ALS_CONT |
                             ISL29023_CMD_I_INT_PERSIST_8),
                            ISL29023AppCallback, &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    ISL29023AppI2CWait(__FILE__, __LINE__);

    //
    // Configure the upper threshold to 80% of maximum value
    //
    g_sISL29023Inst.pui8Data[1] = 0xCC;
    g_sISL29023Inst.pui8Data[2] = 0xCC;
    ISL29023Write(&g_sISL29023Inst, ISL29023_O_INT_HT_LSB,
                  g_sISL29023Inst.pui8Data, 2, ISL29023AppCallback,
                  &g_sISL29023Inst);

    //
    // Wait for transaction to complete
    //
    ISL29023AppI2CWait(__FILE__, __LINE__);

    //
    // Configure the lower threshold to 20% of maximum value
    //
    g_sISL29023Inst.pui8Data[1] = 0x33;
    g_sISL29023Inst.pui8Data[2] = 0x33;
    ISL29023Write(&g_sISL29023Inst, ISL29023_O_INT_LT_LSB,
                  g_sISL29023Inst.pui8Data, 2, ISL29023AppCallback,
                  &g_sISL29023Inst);
    //
    // Wait for transaction to complete
    //
    ISL29023AppI2CWait(__FILE__, __LINE__);

    //
    //Configure and enable SysTick Timer
    //
    MAP_SysTickPeriodSet(ui32SysClock / SYSTICKS_PER_SECOND);
    MAP_SysTickIntEnable();
    MAP_SysTickEnable();

    //
    // Configure PQ4 to control the blue LED.
    //
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTQ_BASE, GPIO_PIN_4);

    //
    // Print label once on the LCD.
    //
    GrStringDraw(&g_sContext, "Visible Lux", 11,
                 ((GrContextDpyWidthGet(&g_sContext) / 2) - 80),
                 (GrContextDpyHeightGet(&g_sContext) - 32) / 2, 1);

    //
    // Loop Forever
    //
    while(1)
    {
        MAP_SysCtlSleep();

        if(g_vui8DataFlag)
        {
            g_vui8DataFlag = 0;

            //
            // Blink the blue LED to indicate activity.
            //
            MAP_GPIOPinWrite(GPIO_PORTQ_BASE, GPIO_PIN_4,
                             ((GPIOPinRead(GPIO_PORTQ_BASE, GPIO_PIN_4)) ^
                              GPIO_PIN_4));

            //
            // Get a local floating point copy of the latest light data
            //
            ISL29023DataLightVisibleGetFloat(&g_sISL29023Inst, &fAmbient);

            //
            // Perform the conversion from float to a printable set of integers
            //
            i32IntegerPart = (int32_t)fAmbient;
            i32FractionPart = (int32_t)(fAmbient * 1000.0f);
            i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
            if(i32FractionPart < 0)
            {
                i32FractionPart *= -1;
            }

            //
            // Print the temperature as integer and fraction parts to LCD and
            // terminal.
            //
            usnprintf(pcBuf, sizeof(pcBuf), "%3d.%03d   ", i32IntegerPart,
                                                           i32FractionPart);
            GrStringDraw(&g_sContext, pcBuf, 9,
                         ((GrContextDpyWidthGet(&g_sContext) / 2) + 16),
                         (GrContextDpyHeightGet(&g_sContext) - 32) / 2, 1);
            UARTprintf("Visible Lux: %3d.%03d\n", i32IntegerPart,
                       i32FractionPart);

            //
            // Check if the intensity of light has crossed a threshold. If so
            // then adjust range of sensor readings to track intensity.
            //
            if(g_vui8IntensityFlag)
            {
                //
                // Disable the low priority interrupts leaving only the I2C
                // interrupt enabled.
                //
                MAP_IntPriorityMaskSet(0x40);

                //
                // Reset the intensity trigger flag.
                //
                g_vui8IntensityFlag = 0;

                //
                // Adjust the lux range.
                //
                ISL29023AppAdjustRange(&g_sISL29023Inst);

                //
                // Now we must manually clear the flag in the ISL29023
                // register.
                //
                ISL29023Read(&g_sISL29023Inst, ISL29023_O_CMD_I,
                             g_sISL29023Inst.pui8Data, 1, ISL29023AppCallback,
                             &g_sISL29023Inst);

                //
                // Wait for transaction to complete
                //
                ISL29023AppI2CWait(__FILE__, __LINE__);

                //
                // Disable priority masking so all interrupts are enabled.
                //
                MAP_IntPriorityMaskSet(0);
            }
        }
    }
}
