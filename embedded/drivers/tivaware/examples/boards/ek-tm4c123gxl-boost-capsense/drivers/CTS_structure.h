//*****************************************************************************
//
// CTS_structure.h - Capacative Sense structures.
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

#ifndef __CTS_STRUCTURE_H__
#define __CTS_STRUCTURE_H__

//*****************************************************************************
//
// If building with a C++ compiler, make all of the definitions in this header
// have a C binding.
//
//*****************************************************************************
#ifdef __cplusplus
extern "C"
{
#endif

//*****************************************************************************
//
// Definitions for the capsense elements and sensors to be used on the TI
// CapSense Booster pack.
//
//*****************************************************************************
#define MAXIMUM_NUMBER_OF_ELEMENTS_PER_SENSOR \
                                4
#define TOTAL_NUMBER_OF_ELEMENTS \
                                6
#define ILLEGAL_SLIDER_WHEEL_POSITION \
                                0xFFFFFFFF
#define WHEEL

//*****************************************************************************
//
// This structure is used by the CTS_structure.c file to describe the essential
// characteristics of capacitive sensor elements, such as those used on the
// CapSense Booster pack.
//
//*****************************************************************************
typedef struct
{
    //
    // The base address of the GPIO port used to access this element.
    //
    uint32_t ui32GPIOPort;

    //
    // The GPIO number corresponding to the pin attached to this element.
    //
    uint32_t ui32GPIOPin;

    //
    // The capacitance measurement threshold beyond which this element will be
    // considered "active". The units of this number are determined by the
    // units used in the function that performs capacitance measurements.
    // Whether "beyond" means higher or lower is determined by the direction of
    // interest at the CTS_Layer level.
    //
    uint32_t ui32Threshold;

    //
    // The capacitance value at which this element should be considered as
    // delivering its maximum touch response.
    //
    uint32_t ui32MaxResponse;
}
tCapTouchElement;

//*****************************************************************************
//
// This structure is used by the CTS_structure.c file to describe groupings of
// capacitive sensor elements designed to function together (referred to
// collectively as a "sensor").
//
//*****************************************************************************
typedef struct tSensor
{
    //
    // The number of elements in this sensor.
    //
    uint8_t ui8NumElements;

    //
    // The number of cycles to be used for capacitance measurements of elements
    // within this sensor (used with the CapSenseElementSystickRC() function).
    //
    uint32_t ui32NumSamples;

    //
    // If this is a slider or wheel element, this is the number of distinct
    // points to be recognized along the surface of the physical sensor.
    //
    uint8_t ui8Points;

    //
    // For wheels and sliders, this is a threshold that determines how much
    // cumulative normalized response a set of 3 adjacent sensors must have
    // before a position on the wheel/slider should be calculated. This changes
    // the size of the region where the wheel/slider will respond to touches.
    // Lower values correspond to wheels with wider radii, and sliders with
    // more breadth.
    //
    // Valid  values range from 0 to 300.
    //
    uint32_t ui32SensorThreshold;

    //
    // The index in the global g_ui32Baselines array where baseline capacitance
    // measurements for all elements in this sensor are stored.
    //
    uint32_t ui32BaseOffset;

    //
    // The array of elements that make up this sensor.
    //
    const tCapTouchElement *psElement[MAXIMUM_NUMBER_OF_ELEMENTS_PER_SENSOR];
}
tSensor;

//*****************************************************************************
//
// Prototypes for the globals defined in CTS_structure.c
//
//*****************************************************************************
extern const tCapTouchElement g_sVolumeDownElement;
extern const tCapTouchElement g_sVolumeUpElement;
extern const tCapTouchElement g_sRightElement;
extern const tCapTouchElement g_sLeftElement;
extern const tCapTouchElement g_sMiddleElement;
extern const tCapTouchElement g_sProximityElement;
extern const tSensor g_sSensorWheel;
extern const tSensor g_sMiddleButton;

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

#endif // __CTS_STRUCTURE_H__
