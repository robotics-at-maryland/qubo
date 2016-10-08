//*****************************************************************************
//
// CTS_Layer.c - Capacitive Sense Library Hardware Abstraction Layer.
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

#include <stdint.h>
#include <stdbool.h>
#include "drivers/CTS_structure.h"
#include "drivers/CTS_Layer.h"
#include "drivers/CTS_HAL.h"

//*****************************************************************************
//
//! \addtogroup CTS_API
//! @{
//
//*****************************************************************************

//*****************************************************************************
//
// Global variables for sensing
//
//*****************************************************************************
uint32_t g_ui32Baselines[TOTAL_NUMBER_OF_ELEMENTS];
uint32_t g_pui32MeasCount[MAXIMUM_NUMBER_OF_ELEMENTS_PER_SENSOR];
uint32_t g_ui32CtsStatusReg = (DOI_INC + TRADOI_FAST + TRIDOI_SLOW);

//*****************************************************************************
//
//! Measure the capacitance of each element within the Sensor
//!
//! \param psSensor Pointer to Sensor structure to be measured
//!
//! \param pui32Counts Address to where the measurements are to be written
//!
//! This function selects the appropriate HAL to perform the capacitance
//! measurement based upon the halDefinition found in the sensor structure.
//! The order of the elements within the Sensor structure is arbitrary but must
//! be consistent between the application and configuration. The first element
//! in the array (counts) corresponds to the first element within the Sensor
//! structure.
//!
//! \return None.
//
//*****************************************************************************
void
TI_CAPT_Raw(const tSensor *psSensor, uint32_t *pui32Counts)
{
    CapSenseSystickRC(psSensor, pui32Counts);
}

//*****************************************************************************
//
//! Make a single capacitance measurement to initialize baseline tracking.
//!
//! \param  psSensor Pointer to Sensor structure whose baseline capacitance is
//! to be measured.
//!
//! This function calls TI_CAPT_Raw(), and saves the resulting measurement to
//! the main array of baseline capacitance measurements. This should be called
//! towards the beginning of any application that intends to use the more
//! complex measurement functions.
//!
//! \note This function MUST be called on a sensor structure before any other
//! function in this file is called on that same sensor structure, with the
//! exception of TI_CAPT_Raw().
//!
//! \return None.
//
//*****************************************************************************
void
TI_CAPT_Init_Baseline(const tSensor *psSensor)
{
    TI_CAPT_Raw(psSensor,
                &g_ui32Baselines[psSensor->ui32BaseOffset]);
}

//*****************************************************************************
//
//! Update baseline capacitance numbers for a single sensor.
//!
//! \param  psSensor Pointer to Sensor structure whose baseline values are to
//! be measured.
//!
//! \param  ui8NumberOfAverages Number of measurements to be averaged to form
//! the new baseline capacitance.
//!
//! This function takes a series of TI_CAPT_Raw() measurements and averages
//! them in to the baseline capacitance value set by TI_CAPT_Init_Baseline().
//! This function should not be used without first calling
//! TI_CAPT_Init_Baseline().
//!
//! \return none
//
//*****************************************************************************
void
TI_CAPT_Update_Baseline(const tSensor* psSensor, uint8_t ui8NumberOfAverages)
{
    uint8_t ui8Loop2,ui8Loop;

    //
    // For every element in the sensor, loop through ui8NumberOfAverages times,
    // and average the results in to the baseline number stored in the main
    // baseline tracking array.
    //
    for(ui8Loop = 0; ui8Loop < ui8NumberOfAverages; ui8Loop++)
    {
        for(ui8Loop2 = 0; ui8Loop2 < psSensor->ui8NumElements; ui8Loop2++)
        {
            //
            // Take a capacitance measurement.
            //
            TI_CAPT_Raw(psSensor, g_pui32MeasCount);

            //
            // Average into the main baseline array, weighting more recent
            // samples more heavily than older samples.
            //
            g_ui32Baselines[ui8Loop2 + psSensor->ui32BaseOffset] =
                  ((g_pui32MeasCount[ui8Loop2] / 2) +
                   (g_ui32Baselines[ui8Loop2 + psSensor->ui32BaseOffset] / 2));
        }
    }
}

//*****************************************************************************
//
//! Reset the Baseline Tracking algorithm to the default state.
//!
//! This function resets the Baseline Tracking algorithm to use the default
//! settings: an increasing direction of interest, fast tracking of against
//! the direction of interest, and slow tracking of changes in the direction of
//! interest.
//!
//! \return  none
//
//*****************************************************************************
void
TI_CAPT_Reset_Tracking(void)
{
    g_ui32CtsStatusReg = (DOI_INC | TRADOI_FAST | TRIDOI_SLOW);
}

//*****************************************************************************
//
//! Update the Baseline Tracking algorithm Direction of Interest
//!
//! \param   bDirection is a boolean value corresponding to the desired
//! direction of interest.
//!
//! This function may be used to change the direction of interest for the
//! capacitive sensing driver overall. When the \e bDirection parameter is
//! true, the direction of interest is set to "increasing". If it is false, the
//! direction of interest is set to "decreasing".
//!
//! \return  none
//
//*****************************************************************************
void
TI_CAPT_Update_Tracking_DOI(bool bDirection)
{
    if(bDirection)
    {
        g_ui32CtsStatusReg |= DOI_INC;
    }
    else
    {
        g_ui32CtsStatusReg &= ~DOI_INC;
    }
}

//*****************************************************************************
//
//! Update the baseline tracking algorithm tracking rates
//!
//! \param ui8Rate Rate of tracking changes in and against direction of
//! interest.
//!
//! This function sets the baseline tracking rates both in and against the
//! configured direction of interest. The \e ui8Rate parameter is the logical
//! OR of two configuration values, one for each direction.
//!
//! The rate of baseline tracking IN the direction of interest is set with one
//! of the following values: \b TRIDOI_VSLOW, \b TRIDOI_SLOW, \b TRIDOI_MED,
//! \b TRIDOI_FAST.
//!
//! The rate of baseline tracking AGAINST the direction of interest is set with
//! one of the following values: \b TRADOI_VSLOW, \b TRADOI_SLOW,
//! \b TRADOI_MED, \b TRADOI_FAST.
//!
//! \return  none
//
//*****************************************************************************
void
TI_CAPT_Update_Tracking_Rate(uint8_t ui8Rate)
{
  //
  // Clear the old tracking rates.
  //
  g_ui32CtsStatusReg &= ~(TRIDOI_FAST + TRADOI_VSLOW);

  //
  // Update with new tracking rates.
  //
  g_ui32CtsStatusReg |= (ui8Rate & 0xF0);
}

//*****************************************************************************
//
//! Measure the changes in capacitance "delta counts" of the elements of the
//! chosen sensor in the current direction of interest, and report any detected
//! touch events.
//!
//! \param psSensor Pointer to Sensor structure to be measured
//!
//! \param pui32DeltaCnt Address to where the measurements are to be written
//!
//! For every element in the sensor \e psSensor, this function will measure the
//! current capacitance, and calculate the difference between the raw
//! measurement and the most recent baseline capacitance of that element. If
//! this difference is in the current direction of interest (specified by the
//! \e g_ui32CtsStatusReg variable), the absolute value of the difference in
//! capacitance (referred to as a "delta count") will be placed in the provided
//! \e pui32DeltaCnt array. The order of delta counts in \e pui32DeltaCnt will
//! match the order of elements in \e psSensor.
//!
//! If the delta count of any element exceeds the threshold for that element,
//! this function will also set the \b EVNT bit in the \e g_ui32CtsStatusReg
//! variable. This bit may be used by the caller to determine whether any
//! element in \e psSensor is currently being touched.
//!
//! Finally, if none of the delta counts exceed the threshold for their
//! respective elements, this function will use the measured capacitance values
//! to adjust the baseline capacitance values for all elements in the sensor.
//! The rate at which the baseline value changes is specified by the value of
//! \e g_ui32CtsStatusReg.
//!
//! \return  none
//
//*****************************************************************************
void
TI_CAPT_Custom(const tSensor* psSensor, uint32_t* pui32DeltaCnt)
{
    uint8_t ui8Loop;
    uint32_t ui32RawCount, ui32BoundedRawCount, ui32BaseCount;
    uint32_t ui32Threshold, ui32TrackingFactor;

    //
    // Clear any events that might be left in the global status register from
    // previous iterations of this call.
    //
    g_ui32CtsStatusReg &= ~(EVNT);

    //
    // First, gather raw measurements of all elements in the sensor. We will
    // store them in the array called pui32DeltaCnt for no because we can
    // expect this array to be big enough to hold the data from TI_CAPT_Raw(),
    // but please note that they are not really "delta counts" at this time.
    //
    TI_CAPT_Raw(psSensor, &pui32DeltaCnt[0]);

    //
    // Loop over the elements in the sensor
    //
    for (ui8Loop = 0; ui8Loop < (psSensor->ui8NumElements); ui8Loop++)
    {
        //
        // Obtain the relevant data for this element, including its baseline
        // count, threshold, and its most recent raw count.
        //
        ui32RawCount = pui32DeltaCnt[ui8Loop];
        ui32Threshold = (psSensor->psElement[ui8Loop])->ui32Threshold;
        ui32BaseCount = g_ui32Baselines[ui8Loop + psSensor->ui32BaseOffset];

        //
        // Use our direction of interest to figure out how to interpret the
        // relationship between raw count, baseline, and threshold.
        //
        if(!(g_ui32CtsStatusReg & DOI_MASK))
        {
            //
            // If our direction of interest is DECREASING, meaning that
            // DECREASED count values correspond to touch events, we will use
            // the following formula to obtain the delta count.
            //
            if(ui32BaseCount > ui32RawCount)
            {
                pui32DeltaCnt[ui8Loop] = ui32BaseCount - ui32RawCount;
            }
            else
            {
                pui32DeltaCnt[ui8Loop] = 0;
            }

            //
            // Also, create a bounded version of the raw count for use with the
            // baseline tracking algorithm. Limiting the raw count to be no
            // higher than (baseline + threshold) prevents the baseline from
            // changing too quickly.
            //
            if(ui32RawCount > (ui32BaseCount + ui32Threshold))
            {
                ui32BoundedRawCount = ui32BaseCount + ui32Threshold;
            }
            else
            {
                ui32BoundedRawCount = ui32RawCount;
            }
        }
        else if(g_ui32CtsStatusReg & DOI_MASK)
        {
            //
            // If our direction of interest is INCREASING, meaning that
            // INCREASED count values correspond to touch events, we will use
            // the following formula to obtain a delta count.
            //
            if(ui32RawCount > ui32BaseCount)
            {
                pui32DeltaCnt[ui8Loop] = ui32RawCount - ui32BaseCount;
            }
            else
            {
                pui32DeltaCnt[ui8Loop] = 0;
            }

            //
            // Again, create a bounded version of the raw count for use with the
            // baseline tracking algorithm. Limiting the raw count to be no
            // lower than (baseline - threshold) prevents the baseline from
            // changing too quickly.
            //
            if(ui32RawCount < (ui32BaseCount - ui32Threshold))
            {
                ui32BoundedRawCount = ui32BaseCount - ui32Threshold;
            }
            else
            {
                ui32BoundedRawCount = ui32RawCount;
            }
        }

        //
        // At this point, we have determined our delta value for this element,
        // but we still need to look for touch events and update our baseline
        // measurements. This all depends on the delta count, the threshold,
        // and whether an event has been seen recently.
        //
        if(pui32DeltaCnt[ui8Loop] > ui32Threshold)
        {
            //
            // Our delta was above the threshold, which means we have a touch
            // event. Record this in the g_ui32CtsStatusReg so it can be used
            // by the application. There is no need for further baseline
            // tracking for a while after an event, so set the PAST_EVNT flag
            // as well.
            //
            g_ui32CtsStatusReg |= EVNT;
            g_ui32CtsStatusReg |= PAST_EVNT;
        }
        else if(!(g_ui32CtsStatusReg & PAST_EVNT))
        {
            //
            // If we didn't trigger an event above, and we haven't seen an
            // event recently, we need to do baseline tracking. Calculate the
            // correct baseline tracking factor here.
            //
            if(pui32DeltaCnt[ui8Loop] > 0)
            {
                //
                // If we had a positive delta count, we use the IN
                // direction-of-interest tracking factor
                //
                switch(g_ui32CtsStatusReg & TRIDOI_FAST)
                {
                    //
                    // Special case for "very slow": only move in increments of
                    // one.
                    //
                    case TRIDOI_VSLOW:
                        //
                        // A tracking factor of zero means that the averaging
                        // equation won't be used.
                        //
                        ui32TrackingFactor = 0;

                        //
                        // Move the base count one value towards the bounded
                        // raw count.
                        //
                        if(ui32BoundedRawCount > ui32BaseCount)
                        {
                            ui32BaseCount++;
                        }
                        else
                        {
                            ui32BaseCount--;
                        }
                        break;

                    case TRIDOI_SLOW:
                        ui32TrackingFactor = 128;
                        break;

                    case TRIDOI_MED:
                        ui32TrackingFactor = 4;
                        break;

                    case TRIDOI_FAST:
                        ui32TrackingFactor = 2;
                        break;
                    default:
                        //
                        // We shouldn't ever get here, but this case should
                        // keep us safe.
                        //
                        ui32TrackingFactor = 0;
                }
            }
            else
            {
                //
                // If we had a zero or negative delta count, set the delta
                // count to zero, and use the AGAINST direction-of-interest
                // factor.
                //
                pui32DeltaCnt[ui8Loop] = 0;

                switch ((g_ui32CtsStatusReg & TRADOI_VSLOW))
                {
                    case TRADOI_FAST:
                        ui32TrackingFactor = 2;
                        break;
                    case TRADOI_MED:
                        ui32TrackingFactor = 4;
                        break;
                    case TRADOI_SLOW:
                        ui32TrackingFactor = 64;
                        break;
                    case TRADOI_VSLOW:
                        ui32TrackingFactor = 128;
                        break;
                    default:
                        //
                        // We shouldn't ever get here, but this case should
                        // keep us safe.
                        //
                        ui32TrackingFactor = 0;
                }

            }

            //
            // Update our baseline using our bounded raw count and our tracking
            // factor, and write our new calculated baseline back to the global
            // array.
            //
            if(ui32TrackingFactor)
            {
                ui32BaseCount = ((ui32BaseCount * (ui32TrackingFactor - 1) +
                                  ui32BoundedRawCount) / (ui32TrackingFactor));
            }

            g_ui32Baselines[ui8Loop + psSensor->ui32BaseOffset] = ui32BaseCount;

        }
    }

    //
    // If we got all the way down here without seeing any events, we should
    // clear the PAST_EVNT flag to avoid interfering with future calls of this
    // function.
    //
    if(!(g_ui32CtsStatusReg & EVNT))
    {
        g_ui32CtsStatusReg &= ~PAST_EVNT;
    }
}
//*****************************************************************************
//
//! Determine if any element in the given sensor is being pressed.
//!
//! \param psSensor pointer to the sensor structure whose elements are to be
//! scanned for button presses.
//!
//! This function takes measrements of all elements in \e psSensor and checks
//! to see if any element has exceeded its threshold for a touch event. This
//! function can be used with single-element sensors to create a capacitive
//! "button". The return value is a simple 1 or 0 indication of whether the
//! button is currently being pressed.
//!
//! Please note that this function will also update the baseline capacitance
//! for all elements in the sensor psSensor.
//!
//! \return  Indication if button is pressed (1) or is not being pressed (0)
//
//*****************************************************************************
uint8_t
TI_CAPT_Button(const tSensor *psSensor)
{
    uint8_t ui8Result;

    //
    // Start with the assumption that no button was pressed.
    //
    ui8Result = 0;

    //
    // Perform a quick measurement of the delta counts on the chosen sensor.
    //
    TI_CAPT_Custom(psSensor, g_pui32MeasCount);

    //
    // Check to see if any threshold-crossing events were recorded.
    //
    if(g_ui32CtsStatusReg & EVNT)
    {
        //
        // If so, the button is being pressed. Return a 1. Otherwise return a
        // zero.
        //
        ui8Result = 1;
    }

    return ui8Result;
}

//*****************************************************************************
//
//! Determine which element in a multi-element sensor is being pressed, if any.
//!
//! \param psSensor Pointer to sensor containing elements to be scanned.
//!
//! This function takes a capacitive measurement of all elements in \e
//! psSensor, and checks to see if any element has exceeded the threshold for a
//! touch event. If so, it will will return a pointer to the most active
//! element, normalized by the \e ui32MaxResponse for each element. This
//! function can be used with multi-element sensors to create an array of
//! mutually exclusive buttons.
//!
//! Please note that this function will also update the baseline capacitance
//! for all elements in the sensor psSensor.
//!
//! \return Returns a pointer to the element (button) being pressed or 0 if no
//! element was pressed.
//
//*****************************************************************************
const tCapTouchElement *
TI_CAPT_Buttons(const tSensor *psSensor)
{
    uint8_t ui8Index;

    //
    // Find the delta counts of all elements in the given sensor.
    //
    TI_CAPT_Custom(psSensor, g_pui32MeasCount);

    //
    // If at least one element was pressed, find the dominant element
    // (normalized by maximum response), and return a pointer to that element
    // back to the caller.
    //
    if(g_ui32CtsStatusReg & EVNT)
    {
        ui8Index = Dominant_Element(psSensor, g_pui32MeasCount);

        return(psSensor->psElement[ui8Index]);
    }
    else
    {
        //
        // If there were no touch events at all, return a zero.
        //
        return(0);
    }
}

#ifdef SLIDER
//*****************************************************************************
//
//! Detect touch events on a slider element, and return the position of any
//! touch event found in units of points.
//!
//! \param psSensor Pointer to the slider element to be measured.
//!
//! This function performs a capacitive measurement on the given sensor \e
//! psSensor, and interprets the results assuming that this sensor represents a
//! physical slider. Its return value is a numerical position along the surface
//! of the slider. A position of zero represents a touch event on the extreme
//! end of the slider closest to element zero. A position value matching the
//! value of \e ui8Points for this sensor represents a touch event on the
//! extreme end of the slider corresponding to the last element in the sensor.
//!
//! Please note that this function will also update the baseline capacitance
//! for all elements in the sensor \e psSensor.
//!
//! \return  Returns the calculate position of the touch event on the slider or
//! an illegal value \b ILLEGAL_SLIDER_WHEEL_POSITION if no touch was detected.
//
//*****************************************************************************
uint32_t
TI_CAPT_Slider(const tSensor *psSensor)
{
    uint8_t ui8NumElements, ui8Points, ui8PointsPerElement;
    uint8_t ui8Index;
    uint32_t ui32SensorThreshold, ui32SliderPosition;
    uint32_t ui32ThresholdCheck;

    //
    // Gather important sensor-level information
    //
    ui32SensorThreshold = psSensor->ui32SensorThreshold;
    ui8NumElements = psSensor->ui8NumElements;
    ui8Points = psSensor->ui8Points;
    ui8PointsPerElement = ui8Points / ui8NumElements;

    //
    // Take a measurement of delta counts, and store it in our global
    // measurement array.
    //
    TI_CAPT_Custom(psSensor, g_puiMeasCount);

    //
    // Check the global EVNT flag to see if any elements in this sensor are
    // active. If not, we can skip the calculations and simply return an
    // illegal position.
    //
    if(g_ui32CtsStatusReg & EVNT)
    {
        //
        // If we did have a touch event, normalize the responses and find the
        // index of the dominant element.
        //
        ui8Index = Dominant_Element(psSensor, &g_pui32MeasCount[0]);

        //
        // Check to see if the normalized responses of the dominant element and
        // the adjacent elements are collectively high enough to cross the
        // overall sensor threshold. If so, we can conclude that the touch
        // event is somewhere within the intended track of the physical slider,
        // and we can go ahead and calculate the position.  Make sure to handle
        // the first and last elements carefully here, as they have fewer
        // neighbors.
        //
        if(ui8Index == 0)
        {
            ui32ThresholdCheck = (g_pui32MeasCount[ui8Index] +
                                  g_pui32MeasCount[ui8Index + 1]);
        }
        else if(ui8Index == (ui8NumElements - 1))
        {
            ui32ThresholdCheck = (g_pui32MeasCount[ui8Index] +
                                  g_pui32MeasCount[ui8Index - 1]);
        }
        else
        {
            ui32ThresholdCheck = (g_pui32MeasCount[ui8Index] +
                                  g_pui32MeasCount[ui8Index + 1] +
                                  g_pui32MeasCount[ui8Index - 1]);
        }

        //
        // If we didn't pass our threshold check, we probably have a touch
        // event close to the sensor, but not actually in the desired region of
        // the physical slider. This means we should stop our calculation here
        // and return an illegal position value.
        //
        if(ui32ThresholdCheck < ui32SensorThreshold)
        {
            return ILLEGAL_SLIDER_WHEEL_POSITION;
        }

        //
        // If we passed the check, it's time to calculate the posistion (in
        // points) of the touch.  We will start with the assumption that the
        // touch is in the exact center of the dominant element
        //
        ui32SliderPosition = ((ui8Index * ui8PointsPerElement) +
                              (ui8PointsPerElement / 2));

        //
        // Then we will improve our calculation of the touch position by
        // factoring in the measurements from the two adjacent elements. The
        // first and last sensors in the slider are special cases, as each of
        // them only has one adjacent element.
        //
        if(ui8Index == 0)
        {
            //
            // Special case for the first element in the array. If the adjacent
            // sensor is responding, push the position toward that element
            // accordingly. Otherwise, push the position away from that element
            // based on the magnitude of the response from sensor zero.
            //
            if(g_pui32MeasCount[ui8Index + 1])
            {
                ui32SliderPosition += ((g_pui32MeasCount[ui8Index + 1] *
                                        ui8PointsPerElement) / 100);
            }
            else
            {
                ui32SliderPosition = ((g_pui32MeasCount[ui8Index] *
                                       ui8PointsPerElement) / 200);
            }
        }
        else if(ui8Index == (ui8NumElements - 1))
        {
            //
            // Special case for the last element in the array. If the adjacent
            // sensor is responding, push the position toward that element
            // accordingly. Otherwise, push the position away from that element
            // based on the magnitude of the response of the last element.
            //
            if(g_pui32MeasCount[ui8Index - 1])
            {
                ui32SliderPosition -= ((g_pui32MeasCount[ui8Index - 1] *
                                        ui8PointsPerElement) / 100);
            }
            else
            {
                ui32SliderPosition = ui8Points;
                ui32SliderPosition -= ((g_pui32MeasCount[ui8Index] *
                                        ui8PointsPerElement) / 200);
            }
        }
        else
        {
            //
            // All other elements will have two neighbors, so push the position
            // towards based on the response of the adjacent sensors only.
            //
            ui32SliderPosition += ((g_pui32MeasCount[ui8Index + 1] *
                                    ui8PointsPerElement) / 100);

            ui32SliderPosition -= ((g_pui32MeasCount[ui8Index - 1] *
                                    ui8PointsPerElement) / 100);
        }

        //
        // Return the adjusted position back to the caller.
        //
        return ui32SliderPosition;

    }
    else
    {
        //
        // We didn't register any touch events at all, so return an illegal
        // slider position
        //
        return ILLEGAL_SLIDER_WHEEL_POSITION;
    }
}
#endif // SLIDER

#ifdef WHEEL
//*****************************************************************************
//
//! Detect touch events on a wheel element, and return the position of any
//! touch event found in units of points.
//!
//! \param psSensor Pointer to the wheel element to be measured.
//!
//! This function performs a capacitive measurement on the given sensor \e
//! psSensor, and interprets the results assuming that this sensor represents a
//! physical wheel. Its return value is a numerical position along the surface
//! of the wheel. A position of zero represents a touch on the wheel centered
//! at the point where the element of index zero and the last element in the
//! array physically touch. Position values increase around the circumference
//! of the wheel in the same direction as increasing element indices in the \e
//! psSensor structure.
//!
//! Please note that this function will also update the baseline capacitance
//! for all elements in the sensor \e psSensor.
//!
//! \return  Returns the calculate position of the touch event on the wheel or
//! an illegal value \b ILLEGAL_SLIDER_WHEEL_POSITION if no touch was detected.
//
//*****************************************************************************
uint32_t
TI_CAPT_Wheel(const tSensor* psSensor)
{
    uint8_t ui8NumElements, ui8Points, ui8PointsPerElement;
    uint8_t ui8Index;
    uint32_t ui32SensorThreshold, ui32WheelPosition;
    uint32_t ui32ThresholdCheck;

    //
    // Gather important sensor-level information
    //
    ui32SensorThreshold = psSensor->ui32SensorThreshold;
    ui8NumElements = psSensor->ui8NumElements;
    ui8Points = psSensor->ui8Points;
    ui8PointsPerElement = ui8Points / ui8NumElements;

    //
    // Take a measurement of delta counts, and store it in our global
    // measurement array.
    //
    TI_CAPT_Custom(psSensor, g_pui32MeasCount);

    //
    // Check the global EVNT flag to see if any elements in this sensor are
    // active. If not, we can skip the calculations and simply return an
    // illegal position.
    //
    if(g_ui32CtsStatusReg & EVNT)
    {
        //
        // If we did have a touch event, normalize the responses and find the
        // index of the dominant element.
        //
        ui8Index = Dominant_Element(psSensor, &g_pui32MeasCount[0]);

        //
        // Check to see if the normalized responses of the dominant element and
        // the adjacent elements are collectively high enough to cross the
        // overall sensor threshold. If so, we can conclude that the touch
        // event is somewhere within the intended track of the physical wheel,
        // and we can go ahead and calculate the position.  Make sure to handle
        // the first and last elements carefully, as their neighbors wrap
        // around the array boundary.
        //
        if(ui8Index == 0)
        {
            ui32ThresholdCheck = (g_pui32MeasCount[ui8Index] +
                                  g_pui32MeasCount[ui8Index + 1] +
                                  g_pui32MeasCount[ui8NumElements - 1]);
        }
        else if(ui8Index == (ui8NumElements - 1))
        {
            ui32ThresholdCheck = (g_pui32MeasCount[ui8Index] +
                                  g_pui32MeasCount[ui8Index - 1] +
                                  g_pui32MeasCount[0]);
        }
        else
        {
            ui32ThresholdCheck = (g_pui32MeasCount[ui8Index] +
                                  g_pui32MeasCount[ui8Index + 1] +
                                  g_pui32MeasCount[ui8Index - 1]);
        }

        //
        // If we didn't pass our threshold check, we probably have a touch
        // event close to the sensor, but not actually in the desired region of
        // the physical wheel. This means we should stop our calculation here
        // and return an illegal position value.
        //
        if(ui32ThresholdCheck < ui32SensorThreshold)
        {
            return ILLEGAL_SLIDER_WHEEL_POSITION;
        }

        //
        // If we passed the check, it's time to calculate the position (in
        // points) of the touch.  We will start with the assumption that the
        // touch is in the exact center of the dominant element
        //
        ui32WheelPosition = ((ui8Index * ui8PointsPerElement) +
                             (ui8PointsPerElement / 2));

        //
        // Then we will improve our calculation of the touch position by
        // factoring in the measurements from the two adjacent elements. The
        // first and last sensors in the wheel are special cases, as each of
        // them only has one adjacent element.
        //
        if(ui8Index == 0)
        {
            //
            // Special case for the first element in the array, which requires
            // wrapping of the index.
            //
            ui32WheelPosition += ((g_pui32MeasCount[ui8Index + 1] *
                                   ui8PointsPerElement) / 100);

            ui32WheelPosition -= ((g_pui32MeasCount[ui8NumElements - 1] *
                                   ui8PointsPerElement) / 100);
        }
        else if(ui8Index == (ui8NumElements - 1))
        {
            //
            // Special case for the last element in the array, which requires
            // wrapping of the index.
            //
            ui32WheelPosition += ((g_pui32MeasCount[0] *
                                   ui8PointsPerElement) / 100);

            ui32WheelPosition -= ((g_pui32MeasCount[ui8Index - 1] *
                                   ui8PointsPerElement) / 100);
        }
        else
        {
            //
            // No wrapping necessary, so just push the position based on the
            // measurements of the adjacent elements
            //
            ui32WheelPosition += ((g_pui32MeasCount[ui8Index + 1] *
                                   ui8PointsPerElement) / 100);

            ui32WheelPosition -= ((g_pui32MeasCount[ui8Index - 1] *
                                   ui8PointsPerElement) / 100);
        }

        //
        // Return the adjusted position back to the caller.
        //
        return ui32WheelPosition;

    }
    else
    {
        //
        // We didn't register any touch events at all, so return an illegal
        // wheel position
        //
        return ILLEGAL_SLIDER_WHEEL_POSITION;
    }
}
#endif // WHEEL

//*****************************************************************************
//
//! Normalize a set of delta counts for the given sensor, and return the index
//! of the element with the highest normalized response.
//!
//! \param psSensor Pointer to the sensor to be evaluated.
//!
//! \param pui32DeltaCnt Pointer to an array of delta counts corresponding to
//! the sensor being evaluated.
//!
//! This function attempts to find the element of \e psSensor with the highest
//! normalized response based on a prior TI_CAPT_Custom() measurement. This
//! function assumes that it is being called after a touch event has occurred,
//! meaning that the \b EVNT bit is set, and at least one element in the sensor
//! structure has a delta count value greater than its threshold.
//!
//! Please be aware that this function alters the \e pui32DeltaCnt array,
//! replacing the delta counts with a percentage value for each element. These
//! percentages are the values used to determine which element has the highest
//! response. This percentage is calculated by the following formula:
//!
//! \verbatim
//! DeltaCount = 100 * ((DeltaCount - Threshold) / (MaxResponse - Threshold))
//! \endverbatim
//!
//! \return Returns the index of the element in \e psSensor whose normalized
//! response is the highest. Returns a default value of 0 if all elements are
//! below threshold.
//
//*****************************************************************************
uint8_t
Dominant_Element(const tSensor *psSensor, uint32_t* pui32DeltaCnt)
{
    uint8_t ui8Loop, ui8DominantElement;
    uint32_t ui32ElementThreshold, ui32ElementMaxResponse;
    uint32_t ui32DynamicRange, ui32AdjDelta;
    uint32_t ui32HighestNormalizedDelta;

    //
    // Start with the assumption that we have no dominant element, and
    // initialize our highest normalized delta value to zero.
    //
    ui8DominantElement = 0;
    ui32HighestNormalizedDelta = 0;

    //
    // Loop over all elements in this sensor. Normalize the delta count of each
    // element, and keep track of the element with the highest normalized
    // response.
    //
    for(ui8Loop = 0; ui8Loop < psSensor->ui8NumElements; ui8Loop++)
    {
        //
        // Retrieve the threshold and maximum response for this element, and
        // calculate the full dynamic range.
        //
        ui32ElementThreshold = (psSensor->psElement[ui8Loop])->ui32Threshold;

        ui32ElementMaxResponse = ((psSensor->psElement[ui8Loop])->
                                  ui32MaxResponse);

        ui32DynamicRange = ui32ElementMaxResponse - ui32ElementThreshold;

        //
        // Check to see if this element's delta count is over the threshold.
        // If so, further calculations are necessary. If not, the normalized
        // value is zero.
        //
        if(pui32DeltaCnt[ui8Loop] >= ui32ElementThreshold)
        {
            //
            // Limit the delta count for this element to it's maximum response.
            //
            if(pui32DeltaCnt[ui8Loop] > (ui32ElementMaxResponse))
            {
                pui32DeltaCnt[ui8Loop] = ui32ElementMaxResponse;
            }

            //
            // Calculate an adjusted delta for the normalized response.
            //
            ui32AdjDelta = pui32DeltaCnt[ui8Loop] - ui32ElementThreshold;

            //
            // Convert the value in the delta count in the given array to a
            // percentage of this element's dynamic range.
            //
            pui32DeltaCnt[ui8Loop] = ((100 * ui32AdjDelta) /
                                      ui32DynamicRange);

            //
            // If this normalized delta count is the highest seen so far,
            // record the index of the element.
            //
            if(pui32DeltaCnt[ui8Loop] > ui32HighestNormalizedDelta)
            {
                ui32HighestNormalizedDelta = pui32DeltaCnt[ui8Loop];
                ui8DominantElement = ui8Loop;
            }
        }
        else
        {
            pui32DeltaCnt[ui8Loop] = 0;
        }
    }

    //
    // Return the index of the element with the highest normalized response.
    //
    return(ui8DominantElement);
}

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************
