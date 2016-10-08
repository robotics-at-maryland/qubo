//*****************************************************************************
//
// boostxl_battpack.c - Example for use with the Fuel Tank BoosterPack
//                      (BOOSTXL-BATTPACK)
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

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "inc/hw_gpio.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "driverlib/uart.h"
#include "utils/ustdlib.h"
#include "utils/cmdline.h"
#include "utils/uartstdio.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/bq27510g3.h"
#include "sensorlib/hw_bq27510g3.h"
#include "drivers/rgb.h"


//*****************************************************************************
//
//! \addtogroup example_list
//! <h1>Fuel Tank BoosterPack Measurement example application (boostxl_battpack)</h1>
//!
//! This example demonstrates the basic use of the Sensor Library, TM4C123G
//! LaunchPad and the Fuel Tank BoosterPack to obtain state-of-charge,
//! battery voltage, temperature, and several other supported  measurements
//! via the BQ27510G3 gas gauge sensor on the Fuel tank boosterpack.
//!
//! Connect a serial terminal program to the LaunchPad's ICDI virtual serial
//! port at 115,200 baud.  Use eight bits per byte, no parity and one stop bit.
//! The raw sensor measurements are printed to the terminal.
//
//*****************************************************************************

//*****************************************************************************
//
// Define BQ27510G3 I2C Address.
//
//*****************************************************************************
#define BQ27510G3_I2C_ADDRESS      0x55

//*****************************************************************************
//
// Global array for holding the color values for the RGB.
//
//*****************************************************************************
uint32_t g_pui32Colors[3];

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Global instance structure for the BQ27510G3 sensor driver.
//
//*****************************************************************************
tBQ27510G3 g_sBQ27510G3Inst;

//*****************************************************************************
//
// Global new data flag to alert main that BQ27510G3 data is ready.
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
// Defines the size of the buffer that holds the command line.
//
//*****************************************************************************
#define CMD_BUF_SIZE    64

//*****************************************************************************
//
// Global buffer that holds the command line.
//
//*****************************************************************************
static char g_cCmdBuf[CMD_BUF_SIZE];

//*****************************************************************************
//
// BQ27510G3 Sensor callback function.  Called at the end of BQ27510G3 sensor
// driver transactions. This is called from I2C interrupt context. Therefore,
// we just set a flag and let main do the bulk of the computations and display.
//
//*****************************************************************************
void BQ27510G3AppCallback(void* pvCallbackData, uint_fast8_t ui8Status)
{
    if(ui8Status == I2CM_STATUS_SUCCESS)
    {
        //
        // If I2C transaction is successful, set data ready flag.
        //
        g_vui8DataFlag = 1;
    }
    else
    {
        //
        // If I2C transaction fails, set error flag.
        //
    	g_vui8ErrorFlag = ui8Status;
    }
}

//*****************************************************************************
//
// Called by the NVIC as a result of I2C8 Interrupt. I2C8 is the I2C connection
// to the BQ27510G3 fuel guage.
//
// This handler is installed in the vector table for I2C8 by default.  To use
// the Fuel Tank on BoosterPack 1 interface change the startup file to place
// this interrupt in I2C7 vector location.
//
//*****************************************************************************
void
BQ27510G3I2CIntHandler(void)
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
// Function to wait for the BQ27510G3 transactions to complete.
//
//*****************************************************************************
uint_fast8_t
BQ27510G3AppI2CWait(void)
{
	uint_fast8_t ui8RC = 0;

    //
    // Put the processor to sleep while we wait for the I2C driver to
    // indicate that the transaction is complete.
    //
    while((g_vui8DataFlag == 0) && (g_vui8ErrorFlag ==0))
    {
        //
        // Wait for I2C Transactions to complete.
        //
    }

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ErrorFlag)
    {
        UARTprintf("\033[31;1m");
        UARTprintf("I2C Error!\n\n");
        UARTprintf("Please ensure FuelTank BP is installed correctly\n");
        UARTprintf("and is sufficiently charged for minimum functionality.\n");
        UARTprintf("\033[0m");
        ui8RC = g_vui8ErrorFlag;
    }

    //
    // Clear the data flags for next use.
    //
    g_vui8DataFlag = 0;
    g_vui8ErrorFlag = 0;

    //
    // Return original error code to caller.
    //
    return ui8RC;
}

//*****************************************************************************
//
// Prompts the user for a command, and blocks while waiting for the user's
// input. This function will return after the execution of a single command.
//
//*****************************************************************************
void
CheckForUserCommands(void)
{
    int iStatus;

    //
    // Peek to see if a full command is ready for processing
    //
    if(UARTPeek('\r') == -1)
    {
        //
        // If not, return so other functions get a chance to run.
        //
        return;
    }

    //
    // If we do have commands, process them immediately in the order they were
    // received.
    //
    while(UARTPeek('\r') != -1)
    {
        //
        // Get a user command back
        //
        UARTgets(g_cCmdBuf, sizeof(g_cCmdBuf));

        //
        // Process the received command
        //
        iStatus = CmdLineProcess(g_cCmdBuf);

        //
        // Handle the case of bad command.
        //
        if(iStatus == CMDLINE_BAD_CMD)
        {
            UARTprintf("Bad command! Type 'h' for help!\n");
        }

        //
        // Handle the case of too many arguments.
        //
        else if(iStatus == CMDLINE_TOO_MANY_ARGS)
        {
            UARTprintf("Too many arguments for command processor!\n");
        }
    }

    //
    // Print the prompt
    //
    UARTprintf("\nBattpack> ");
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
// This function implements the "Cmd_temp" command.
// This function displays the current temp of the battery boosterpack
//
//*****************************************************************************
int
Cmd_temp(int argc, char *argv[])
{
    float fTemperature;
    uint_fast8_t ui8Status;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;

    UARTprintf("\nPolling Temperature, press any key to stop\n\n");

    while(1)
    {
    	//
        // Detect if any key has been pressed.
        //
        if(UARTRxBytesAvail() > 0)
        {
        	UARTFlushRx();
        	break;
        }

		//
        // Get the raw data from the sensor over the I2C bus.
		//
        BQ27510G3DataRead(&g_sBQ27510G3Inst, BQ27510G3AppCallback,
                          &g_sBQ27510G3Inst);

	    //
	    // Wait for callback to indicate request is complete.
	    //
	    ui8Status = BQ27510G3AppI2CWait();

        //
        // If initial read is not successful, exit w/o trying other reads.
	    //
	    if(ui8Status != 0)
	    {
	    	return(0);
	    }

	    //
        // Call routine to retrieve data in float format.
	    //
        BQ27510G3DataTemperatureInternalGetFloat(&g_sBQ27510G3Inst,
                                                 &fTemperature);

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
        // Print temperature with one digit of decimal precision.
        //
        UARTprintf("Current Temperature: \t%3d.%01d C\t\t", i32IntegerPart, 
                i32FractionPart);

		//
		// Print new line.
		//
		UARTprintf("\n");

		//
		// Delay for one second. This is to keep sensor duty cycle
		// to about 10% as suggested in the datasheet, section 2.4.
		// This minimizes self heating effects and keeps reading more accurate.
		//
		ROM_SysCtlDelay(ROM_SysCtlClockGet() / 3);

    }

    //
	// Return success.
	//
	return(0);

}

//*****************************************************************************
//
// This function implements the "Cmd_volt" command.
// This function displays the current battery voltage of the battery
// boosterpack.
//
//*****************************************************************************
int
Cmd_volt(int argc, char *argv[])
{
    float fData;
    uint_fast8_t ui8Status;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;

    UARTprintf("\nPolling battery voltage, press any key to stop\n\n");

    while(1)
    {
    	//
        // Detect if any key has been pressed.
        //
        if(UARTRxBytesAvail() > 0)
        {
        	UARTFlushRx();
        	break;
        }

		//
        // Get the raw data from the sensor over the I2C bus.
		//
        BQ27510G3DataRead(&g_sBQ27510G3Inst, BQ27510G3AppCallback,
                          &g_sBQ27510G3Inst);

	    //
	    // Wait for callback to indicate request is complete.
	    //
	    ui8Status = BQ27510G3AppI2CWait();

        //
        // If initial read is not successful, exit w/o trying other reads.
	    //
	    if(ui8Status != 0)
	    {
	    	return(0);
	    }

	    //
        // Call routine to retrieve data in float format.
	    //
        BQ27510G3DataVoltageBatteryGetFloat(&g_sBQ27510G3Inst, &fData);

		//
		// Convert the data to an integer part and fraction part for easy
		// print.
		//
		i32IntegerPart = (int32_t) fData;
		i32FractionPart =(int32_t) (fData * 1000.0f);
		i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
		if(i32FractionPart < 0)
		{
			i32FractionPart *= -1;
		}

        //
        // Print voltage with one digit of decimal precision.
        //
        UARTprintf("Battery Voltage: \t%3d.%01d V\t\t", i32IntegerPart, 
                i32FractionPart);

		//
		// Print new line.
		//
		UARTprintf("\n");

		//
		// Delay for one second. This is to keep sensor duty cycle
		// to about 10% as suggested in the datasheet, section 2.4.
		// This minimizes self heating effects and keeps reading more accurate.
		//
		ROM_SysCtlDelay(ROM_SysCtlClockGet() / 3);

    }

    //
    // Return success.
    //
    return(0);
}

//*****************************************************************************
//
// This function implements the "Cmd_current" command.
// This function displays the battery current to/from the battery boosterpack
//
//*****************************************************************************
int
Cmd_current(int argc, char *argv[])
{
    float fData;
    uint_fast8_t ui8Status;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;

    UARTprintf("\nPolling battery current, press any key to stop\n\n");

    while(1)
    {
    	//
        // Detect if any key has been pressed.
        //
        if(UARTRxBytesAvail() > 0)
        {
        	UARTFlushRx();
        	break;
        }

		//
		// Get the raw data from the sensor over the I2C bus
		//
        BQ27510G3DataRead(&g_sBQ27510G3Inst, BQ27510G3AppCallback,
                          &g_sBQ27510G3Inst);

	    //
	    // Wait for callback to indicate request is complete.
	    //
	    ui8Status = BQ27510G3AppI2CWait();

        //
        // If initial read is not successful, exit w/o trying other reads.
	    //
	    if(ui8Status != 0)
	    {
	    	return(0);
	    }

	    //
        // Call routine to retrieve data in float format.
	    //
        BQ27510G3DataCurrentInstantaneousGetFloat(&g_sBQ27510G3Inst, &fData);
        fData *= 1000.0f;

		//
		// Convert the data to an integer part and fraction part for easy
		// print.
		//
		i32IntegerPart = (int32_t) fData;
		i32FractionPart =(int32_t) (fData * 1000.0f);
		i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
		if(i32FractionPart < 0)
		{
			i32FractionPart *= -1;
		}

        //
        // Print voltage with one digit of decimal precision.
        //
        UARTprintf("Battery Current: \t%3d.%01d mA\t\t", i32IntegerPart, 
                i32FractionPart);

		//
		// Print new line.
		//
		UARTprintf("\n");

		//
		// Delay for one second. This is to keep sensor duty cycle
		// to about 10% as suggested in the datasheet, section 2.4.
		// This minimizes self heating effects and keeps reading more accurate.
		//
		ROM_SysCtlDelay(ROM_SysCtlClockGet() / 3);

    }

    //
    // Return success.
    //
    return(0);
}

//*****************************************************************************
//
// This function implements the "Cmd_charge" command.
// This function displays the remaining charge of the battery boosterpack
//
//*****************************************************************************
int
Cmd_charge(int argc, char *argv[])
{
    float fData;
    uint_fast8_t ui8Status;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;

    UARTprintf("\nPolling remaining charge, press any key to stop\n\n");

    while(1)
    {
    	//
        // Detect if any key has been pressed.
        //
        if(UARTRxBytesAvail() > 0)
        {
        	UARTFlushRx();
        	break;
        }

		//
        // Get the raw data from the sensor over the I2C bus.
		//
        BQ27510G3DataRead(&g_sBQ27510G3Inst, BQ27510G3AppCallback,
                          &g_sBQ27510G3Inst);

	    //
	    // Wait for callback to indicate request is complete.
	    //
	    ui8Status = BQ27510G3AppI2CWait();

        //
        // If initial read is not successful, exit w/o trying other reads.
	    //
	    if(ui8Status != 0)
	    {
	    	return(0);
	    }

	    //
        // Call routine to retrieve data in float format.
	    //
        BQ27510G3DataChargeStateGetFloat(&g_sBQ27510G3Inst, &fData);

		//
		// Convert the data to an integer part and fraction part for easy
		// print.
		//
		i32IntegerPart = (int32_t) fData;
		i32FractionPart =(int32_t) (fData * 1000.0f);
		i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
		if(i32FractionPart < 0)
		{
			i32FractionPart *= -1;
		}

        //
        // Print SoC with one digit of decimal precision.
        //
        UARTprintf("State of Charge: \t%3d.%01d %%\t\t", i32IntegerPart, 
                i32FractionPart);

		//
		// Print new line.
		//
		UARTprintf("\n");

		//
		// Delay for one second. This is to keep sensor duty cycle
		// to about 10% as suggested in the datasheet, section 2.4.
		// This minimizes self heating effects and keeps reading more accurate.
		//
		ROM_SysCtlDelay(ROM_SysCtlClockGet() / 3);

    }

    //
    // Return success.
    //
    return(0);
}
//*****************************************************************************
//
// This function implements the "Cmd_health" command.
// This function displays the current health of the battery boosterpack
//
//*****************************************************************************
int
Cmd_health(int argc, char *argv[])
{
    float fData;
    uint_fast8_t ui8Status;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;

	//
    // Get the raw data from the sensor over the I2C bus.
	//
    BQ27510G3DataRead(&g_sBQ27510G3Inst, BQ27510G3AppCallback,
                      &g_sBQ27510G3Inst);

    //
    // Wait for callback to indicate request is complete.
    //
    ui8Status = BQ27510G3AppI2CWait();

    //
    // If initial read is not successful, exit w/o trying other reads.
    //
    if(ui8Status != 0)
    {
    	return(0);
    }

    //
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataHealthGetFloat(&g_sBQ27510G3Inst, &fData);

	//
	// Convert the data to an integer part and fraction part for easy
	// print.
	//
	i32IntegerPart = (int32_t) fData;
	i32FractionPart =(int32_t) (fData * 1000.0f);
	i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
	if(i32FractionPart < 0)
	{
		i32FractionPart *= -1;
	}

    //
    // Print health with one digit of decimal precision.
    //
    UARTprintf("State of Health: \t%3d.%01d %%\t\t", i32IntegerPart,
            i32FractionPart);

	//
	// Print new line.
	//
	UARTprintf("\n");

    //
    // Return success.
    //
    return(0);
}

//*****************************************************************************
//
// This function implements the "Cmd_status" command.
// This function displays all battery boosterpack status readings at once
//
//*****************************************************************************
int
Cmd_status(int argc, char *argv[])
{
    float fData;
    uint_fast8_t ui8Status;
    int32_t i32IntegerPart;
    int32_t i32FractionPart;
    int32_t i32Current;

    //
    // Get the raw data from the sensor over the I2C bus.
    //
    BQ27510G3DataRead(&g_sBQ27510G3Inst, BQ27510G3AppCallback,
                      &g_sBQ27510G3Inst);

    //
    // Wait for callback to indicate request is complete.
    //
    ui8Status = BQ27510G3AppI2CWait();

    //
    // If initial read is not successful, exit w/o trying other reads.
    //
    if(ui8Status != 0)
    {
    	return(0);
    }

    //
    //*********  Capacity *********
    //

    //
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataCapacityFullAvailableGetFloat(&g_sBQ27510G3Inst, &fData);
    fData *= 1000.0f;

    //
    // Convert the float to an integer part and fraction part for easy
    // print.
    //
    i32IntegerPart = (int32_t) fData;
    i32FractionPart =(int32_t) (fData * 1000.0f);
    i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
    if(i32FractionPart < 0)
    {
        i32FractionPart *= -1;
    }

    //
    // Print capacity with 1 digits of decimal precision.
    //
    UARTprintf("Battery Capacity: \t%3d.%01d mAH\t\t", i32IntegerPart, 
            i32FractionPart);

    //
    // Print new line.
    //
    UARTprintf("\n");

    //
    //********* Remaining Capacity *********
    //

	//
    // Call routine to retrieve data in float format.
	//
    BQ27510G3DataCapacityRemainingGetFloat(&g_sBQ27510G3Inst, &fData);
    fData *= 1000.0f;

	//
	// Convert the float to an integer part and fraction part for easy
	// print.
	//
	i32IntegerPart = (int32_t) fData;
	i32FractionPart =(int32_t) (fData * 1000.0f);
	i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
	if(i32FractionPart < 0)
	{
		i32FractionPart *= -1;
	}

    //
    // Print data with one digit of decimal precision.
    //
    UARTprintf("Remaining Capacity: \t%3d.%01d mAH\t\t", i32IntegerPart,
            i32FractionPart);

	//
	// Print new line.
	//
	UARTprintf("\n");

    //
    //*********  Voltage *********
    //

    //
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataVoltageBatteryGetFloat(&g_sBQ27510G3Inst, &fData);

    //
    // Convert the float to an integer part and fraction part for easy
    // print.
    //
    i32IntegerPart = (int32_t) fData;
    i32FractionPart =(int32_t) (fData * 1000.0f);
    i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
    if(i32FractionPart < 0)
    {
        i32FractionPart *= -1;
    }

    //
    // Print data with 1 digit of decimal precision.
    //
    UARTprintf("Battery Voltage: \t%3d.%01d V\t\t", i32IntegerPart,
            i32FractionPart);

    //
    // Print new line.
    //
    UARTprintf("\n");

    //
    //*********  Temperature *********
    //

    //
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataTemperatureInternalGetFloat(&g_sBQ27510G3Inst, &fData);

    //
    // Convert the float to an integer part and fraction part for easy
    // print.
    //
    i32IntegerPart = (int32_t) fData;
    i32FractionPart =(int32_t) (fData * 1000.0f);
    i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
    if(i32FractionPart < 0)
    {
        i32FractionPart *= -1;
    }

    //
    // Print temperature with one digits of decimal precision.
    //
    UARTprintf("Internal Temperature: \t%3d.%01d C\t\t", i32IntegerPart,
            i32FractionPart);

    //
    // Print new line.
    //
    UARTprintf("\n");

    //
    //********* State of Charge *********
    //

    //
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataChargeStateGetFloat(&g_sBQ27510G3Inst, &fData);

    //
    // Convert the float to an integer part and fraction part for easy
    // print.
    //
    i32IntegerPart = (int32_t) fData;
    i32FractionPart =(int32_t) (fData * 1000.0f);
    i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
    if(i32FractionPart < 0)
    {
        i32FractionPart *= -1;
    }

    //
    // Print data with one digit of decimal precision.
    //
    UARTprintf("State of Charge: \t%3d.%01d %%\t\t", i32IntegerPart, 
            i32FractionPart);

    //
    // Print new line.
    //
    UARTprintf("\n");

    //
    //********* Health *********
    //

	//
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataHealthGetFloat(&g_sBQ27510G3Inst, &fData);

	//
	// Convert the float to an integer part and fraction part for easy
	// print.
	//
	i32IntegerPart = (int32_t) fData;
	i32FractionPart =(int32_t) (fData * 1000.0f);
	i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
	if(i32FractionPart < 0)
	{
		i32FractionPart *= -1;
	}

    //
    // Print data with one digit of decimal precision.
    //
    UARTprintf("State of Health: \t%3d.%01d %%\t\t", i32IntegerPart, 
            i32FractionPart);

	//
	// Print new line.
	//
	UARTprintf("\n");

    //
    //********* Current *********
    //

	//
    // Call routine to retrieve data in float format.
	//
    BQ27510G3DataCurrentAverageGetFloat(&g_sBQ27510G3Inst, &fData);
    fData *= 1000.0f;

	//
	// Convert the float to an integer part and fraction part for easy
	// print.
	//
	i32IntegerPart = (int32_t) fData;
	i32FractionPart =(int32_t) (fData * 1000.0f);
	i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
	if(i32FractionPart < 0)
	{
		i32FractionPart *= -1;
	}

	i32Current = i32IntegerPart;

    //
    // Print data with one digit of decimal precision.
    //
    UARTprintf("Average Current: \t%3d.%01d mA\t\t", i32IntegerPart, 
            i32FractionPart);

	//
	// Print new line.
	//
	UARTprintf("\n");

    //
    //********* Cycles *********
    //

	//
    // Call routine to retrieve data in float format.
    //
    BQ27510G3DataCycleCountGetFloat(&g_sBQ27510G3Inst, &fData);

	//
	// Convert the float to an integer part and fraction part for easy
	// print.
	//
	i32IntegerPart = (int32_t) fData;
	i32FractionPart =(int32_t) (fData * 1000.0f);
	i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
	if(i32FractionPart < 0)
	{
		i32FractionPart *= -1;
	}

	//
	// Print data with one digit of decimal precision.
	//
	UARTprintf("Recharge Cycles: \t%3d cyc\t\t", i32IntegerPart);

	//
	// Print new line.
	//
	UARTprintf("\n");

    //
    //********* Time to Empty *********
    //
	
	//
    // Call routine to retrieve data in float format.
	//
    BQ27510G3DataTimeToEmptyGetFloat(&g_sBQ27510G3Inst, &fData);

	//
	// Convert the float to an integer part and fraction part for easy
	// print.
	//
	i32IntegerPart = (int32_t) fData;
	i32FractionPart =(int32_t) (fData * 1000.0f);
	i32FractionPart = i32FractionPart - (i32IntegerPart * 1000);
	if(i32FractionPart < 0)
	{
		i32FractionPart *= -1;
	}

	//
	// Print data with one digit of decimal precision.
	//
	if(i32IntegerPart > 60)
	{
		UARTprintf("Time Until Empty: \t%3d hrs\t\t", i32IntegerPart/60);
	}
	else if (i32IntegerPart == -1)
	{
		UARTprintf("Time Until Empty: \t   NA\t\t");
	}
	else
	{
		UARTprintf("Time Until Empty: \t%3d min\t\t", i32IntegerPart);
	}

	//
	// Print new line.
	//
	UARTprintf("\n\n");

	if(i32Current > 0)
	{
		UARTprintf("The battery is charging!\r\n");
	}
	else
	{
		UARTprintf("The battery is discharging!\r\n");
	}
    //
    // Return success.
    //
    return(0);
}

//*****************************************************************************
//
// This function implements the "help" command.  It prints a simple list
// of the available commands with a brief description.
//
//*****************************************************************************
int
Cmd_help(int argc, char *argv[])
{
    tCmdLineEntry *pEntry;

    //
    // Print some header text.
    //
    UARTprintf("\nAvailable commands\n");
    UARTprintf("------------------\n");

    //
    // Point at the beginning of the command table.
    //
    pEntry = &g_psCmdTable[0];

    //
    // Enter a loop to read each entry from the command table.  The
    // end of the table has been reached when the command name is NULL.
    //
    while(pEntry->pcCmd)
    {
        //
        // Print the command name and the brief description.
        //
        UARTprintf("%s%s\n", pEntry->pcCmd, pEntry->pcHelp);

        //
        // Advance to the next entry in the table.
        //
        pEntry++;
    }

    //
    // Return success.
    //
    return(0);
}

//*****************************************************************************
//
// This is the table that holds the command names, implementing functions,
// and brief description.
//
//*****************************************************************************
tCmdLineEntry g_psCmdTable[] =
{
    { "help",        Cmd_help, "      : Display list of commands" },
    { "h",           Cmd_help, "         : alias for help" },
    { "?",           Cmd_help, "         : alias for help" },
    { "s",           Cmd_status, "         : alias for status" },
    { "status",      Cmd_status, "    : Display all status" },
    { "health",      Cmd_health, "    : Display health" },
    { "charge",      Cmd_charge, "    : Poll remaining charge" },
    { "temp",        Cmd_temp, "      : Poll temperature" },
    { "volt",        Cmd_volt, "      : Poll battery voltage" },
    { "current",     Cmd_current, "   : Poll battery current" },
    { 0, 0, 0 }
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
// Main 'C' Language entry point.
//
//*****************************************************************************
int
main(void)
{

	//
    // Configure the system frequency.
    //
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_5 | SYSCTL_USE_PLL | SYSCTL_XTAL_16MHZ |
                       SYSCTL_OSC_MAIN); 

    //
    // Initialize the UART.
    //
    ConfigureUART();

    //
    // Clear the terminal and print the welcome message.
    //
	UARTprintf("\033[2J\033[H");
    UARTprintf("Fuel Tank BoosterPack (BQ27510-G3) Example\n");
    UARTprintf("Type 'help' for a list of commands\n");
    UARTprintf("\nBattpack> ");

	//
    // Set the color to a white approximation.
    //
    g_pui32Colors[RED] = 0x8000;
    g_pui32Colors[BLUE] = 0x8000;
    g_pui32Colors[GREEN] = 0x8000;

    //
    // Initialize RGB driver. Use a default intensity and blink rate.
    //
    RGBInit(0);
    RGBColorSet(g_pui32Colors);
    RGBIntensitySet(0.5f);
    RGBEnable();
	
    //
    // The I2C3 peripheral must be enabled before use.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C3);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);

    //
    // Configure the pin muxing for I2C3 functions on port D0 and D1.
    // This step is not necessary if your part does not support pin muxing.
    //
    ROM_GPIOPinConfigure(GPIO_PD0_I2C3SCL);
    ROM_GPIOPinConfigure(GPIO_PD1_I2C3SDA);

    //
    // Select the I2C function for these pins.  This function will also
    // configure the GPIO pins pins for I2C operation, setting them to
    // open-drain operation with weak pull-ups.  Consult the data sheet
    // to see which functions are allocated per pin.
    //
    GPIOPinTypeI2CSCL(GPIO_PORTD_BASE, GPIO_PIN_0);
    ROM_GPIOPinTypeI2C(GPIO_PORTD_BASE, GPIO_PIN_1);
    HWREG(GPIO_PORTD_BASE + GPIO_O_PUR) = 0x3;
	
    //
    //Initial the GPIO for LED
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
    ROM_GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_1);
    ROM_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_1, 0x00);
	
    //
    // Enable interrupts to the processor.
    //
    ROM_IntMasterEnable();

    //
    // Initialize I2C3 peripheral.
    //
    I2CMInit(&g_sI2CInst, I2C3_BASE, INT_I2C3, 0xff, 0xff,
             ROM_SysCtlClockGet());

    //
    // Initialize the BQ27510G3
    //
    BQ27510G3Init(&g_sBQ27510G3Inst, &g_sI2CInst, BQ27510G3_I2C_ADDRESS,
               BQ27510G3AppCallback, &g_sBQ27510G3Inst);

    //
    // Wait for initialization callback to indicate reset request is complete.
    //
    BQ27510G3AppI2CWait();

    //
    // Last thing before the loop start blinking to show we got this far and
    // the tmp006 is setup and ready for auto measure
    //
    RGBBlinkRateSet(1.0f);

    //
    // Begin the data collection and printing.  Loop Forever.
    //
    while(1)
    {
        //
        // Infinite loop to process user commands from prompt
        //
        CheckForUserCommands();

    }//while end
}
