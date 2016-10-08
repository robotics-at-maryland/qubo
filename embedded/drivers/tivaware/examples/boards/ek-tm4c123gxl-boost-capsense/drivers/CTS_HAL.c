//*****************************************************************************
//
// CTS_HAL.c - Capacative Sense Library Hardware Abstraction Layer.
//
// Copyright (c) 2012-2016 Texas Instruments Incorporated.  All rights reserved.
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

//*****************************************************************************
//
//! \addtogroup CTS_API
//! @{
//
//*****************************************************************************

#include <stdint.h>
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"
#include "inc/hw_memmap.h"
#include "inc/hw_nvic.h"
#include "drivers/CTS_structure.h"
#include "drivers/CTS_HAL.h"

//*****************************************************************************
//
//! Measures the capacitance of all capacitive touch sensing elements in a
//! Sensor structure.
//!
//! \param psSensor is a pointer to a Sensor structure containing pointers to
//! the single-pin capacitance-sensing elements for which measurements are to
//! be taken
//!
//! \param pui32Counts is a pointer to the array where the raw capacitance
//! measurements for each element are to be stored.
//!
//! This function assumes that the capacitive sensor is charged by a GPIO, and
//! then discharged through a resistor to GND. The same GPIO used to perform
//! the charging will be used as an input during discharge to sense when the
//! capacitor's voltage has decreased to at least VIL. The function will
//! execute several cycles of this, and report how long the process took in
//! units of SysTick counts. This count should be proportional to the current
//! capacitance of the sensor, and the resistance of the discharge resistor.
//!
//! \return None.
//
//*****************************************************************************
void
CapSenseSystickRC(const tSensor *psSensor, uint32_t *pui32Counts)
{
    uint8_t ui8Index;

    //
    // Run an RC based capacitance measurement routine on each element in the
    // sensor array.
    //
    for(ui8Index = 0; ui8Index < (psSensor->ui8NumElements); ui8Index++)
    {
        //
        // Reset Systick to zero (probably not necessary for most cases, but we
        // want to make sure that Systick doesn't roll over.)
        //
        HWREG(NVIC_ST_CURRENT) = 0x0;

        //
        // Grab a count number from the capacitive sensor element.
        //
        pui32Counts[ui8Index] =
            CapSenseElementSystickRC(psSensor->psElement[ui8Index]->ui32GPIOPort,
                                     psSensor->psElement[ui8Index]->ui32GPIOPin,
                                     psSensor->ui32NumSamples);

    }
}

//*****************************************************************************
//
//! Measures the external capacitance connected to a single GPIO pin using the
//! RC oscillation method.
//!
//! \param ui32GPIOPort is the base address of the GPIO port where the chosen
//! pin is located.
//!
//! \param ui32GPIOPin is the GPIO pin number of the pin whose capacitance is
//! to be measured
//!
//! \param ui32NumSamples is the number of complete charge-and-discharge cycles
//! the pin is to go through for a complete measurement.
//!
//! Given a GPIO port, GPIO pin, and a number of oscillations, this function
//! will provide a measure of the RC time constant of a particular pin. It
//! accomplishes this by driving the pin to digital high (presumably charging
//! an external capacitor in the process), configuring the pin as an input, and
//! waiting for the capacitor to discharge to ground through an external
//! resistor. This process is repeated ui32NumSamples times, and the total
//! duration of the measurement is recorded by SysTick. The total number of
//! "ticks" is returned as a measure of the time constant of the pin. Higher
//! capacitance values will yield higher return values.
//!
//! Please note: This function assumes that SysTick is already configured and
//! running. Also, there is no correction for SysTick wrap-around, so this may
//! need to be handled at the application level.
//!
//! \return Returns the total amount of time required to oscillate the pin for
//! the provided number of cycles. This constitutes a measure of the RC time
//! constant of the pin, which can be used to track the external capacitance
//! attached to this pin.
//
//*****************************************************************************
uint32_t
CapSenseElementSystickRC(uint32_t ui32GPIOPort, uint32_t ui32GPIOPin,
                         uint32_t ui32NumSamples)
{
    uint32_t ui32StartTime;
    uint32_t ui32EndTime;
    uint32_t ui32Index;
    uint32_t ui32GPIODataReg;
    uint32_t ui32GPIODirReg;

    //
    // Save off our GPIO information.
    //
    ui32GPIODataReg = (ui32GPIOPort + (GPIO_O_DATA + (ui32GPIOPin << 2)));
    ui32GPIODirReg = ui32GPIOPort + GPIO_O_DIR;

    //
    // Record the start time
    //
    ui32StartTime = HWREG(NVIC_ST_CURRENT);

    //
    // Loop until we have the requisite number of samples.
    //
    for(ui32Index = 0; ui32Index < ui32NumSamples; ui32Index++)
    {
        //
        // Drive to VDD.
        //
        HWREG(ui32GPIODataReg) = 0xFF;

        //
        // Configure as input
        //
        HWREG(ui32GPIODirReg) = (HWREG(ui32GPIODirReg) & ~(ui32GPIOPin));

        //
        // Wait until the capacitor drains away to ground
        //
        while(HWREG(ui32GPIODataReg))
        {
        }

        //
        // Configure as an output.
        //
        HWREG(ui32GPIODirReg) = (HWREG(ui32GPIODirReg) | ui32GPIOPin);
    }

    //
    // Record the end time, and calculate the difference
    //
    ui32EndTime = HWREG(NVIC_ST_CURRENT);

    //
    // Return the time difference
    //
    return(ui32StartTime - ui32EndTime);
}

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************
