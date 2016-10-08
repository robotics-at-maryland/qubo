//*****************************************************************************
//
// motion.c - Control of calibration, gestures and MPU9150 data aggregation.
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

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/sysctl.h"
#include "utils/uartstdio.h"
#include "sensorlib/hw_mpu9150.h"
#include "sensorlib/hw_ak8975.h"
#include "sensorlib/i2cm_drv.h"
#include "sensorlib/ak8975.h"
#include "sensorlib/mpu9150.h"
#include "sensorlib/comp_dcm.h"
#include "usblib/usblib.h"
#include "usblib/usbhid.h"
#include "drivers/rgb.h"
#include "events.h"
#include "motion.h"

//*****************************************************************************
//
// Global array that contains the colors of the RGB.  Motion is assigned the
// RED LED and should only modify RED.
//
// fast steady blink means I2C bus error.  Power cycle to clear.  usually
// caused by a reset of the system during an I2C transaction causing the slave
// to hold the bus.
//
// quick and brief blinks also occur on each motion system update.
//
//*****************************************************************************
extern volatile uint32_t g_pui32RGBColors[3];

//*****************************************************************************
//
// Global storage to count blink ticks.  This sets blink timing after error.
//
//*****************************************************************************
uint32_t g_ui32RGBMotionBlinkCounter;

//*****************************************************************************
//
// Global instance structure for the I2C master driver.
//
//*****************************************************************************
tI2CMInstance g_sI2CInst;

//*****************************************************************************
//
// Global instance structure for the MPU9150 sensor driver.
//
//*****************************************************************************
tMPU9150 g_sMPU9150Inst;

//*****************************************************************************
//
// Global Instance structure to manage the DCM state.
//
//*****************************************************************************
tCompDCM g_sCompDCMInst;

//*****************************************************************************
//
// Global state variable for the motion state machine.
//
//*****************************************************************************
uint_fast8_t g_ui8MotionState;

//*****************************************************************************
//
// Global flags to alert main that MPU9150 I2C transaction error has occurred.
//
//*****************************************************************************
volatile uint_fast8_t g_vui8ErrorFlag;

//*****************************************************************************
//
// Global storage for most recent Euler angles and Sensor Data
//
//*****************************************************************************
float g_pfEulers[3];
float g_pfAccel[3];
float g_pfGyro[3];
float g_pfMag[3];

//*****************************************************************************
//
// Global storage for previous net acceleration magnitude reading. Helps smooth
// the emit classify readings.
//
//*****************************************************************************
float g_fAccelMagnitudePrevious;

//*****************************************************************************
//
// Global storage Gesture path.
//
//*****************************************************************************
float g_ppfPath[GESTURE_PATH_LENGTH][GESTURE_NUM_STATES];

//*****************************************************************************
//
// Global storage initial state probabilities.
//
//*****************************************************************************
const float g_pfInitProb[GESTURE_NUM_STATES] = {0.994f, 0.001f, 0.001f, 0.001f,
                                                0.001f, 0.001f, 0.001f};

//*****************************************************************************
//
// Global storage probabilities that we may transit from any given state to
// another state.
//
//*****************************************************************************
const float g_ppfTransitionProb[GESTURE_NUM_STATES][GESTURE_NUM_STATES] =
{
    {0.6f,  0.2f, 0.2f, 0.2f, 0.2f, 0.0f,  0.0f},
    {0.1f,  0.9f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
    {0.1f,  0.0f,  0.9f, 0.0f,  0.0f,  0.0f,  0.0f},
    {0.1f,  0.0f,  0.0f,  0.9f, 0.0f,  0.0f,  0.0f},
    {0.1f,  0.0f,  0.0f,  0.0f,  0.7f, 0.1f, 0.1f},
    {0.0f,   0.0f,  0.0f,  0.0f,  0.1f,  0.9f,  0.0f},
    {0.0f,   0.0f,  0.0f,  0.0f,  0.1f,  0.0f,  0.9f}
};

//*****************************************************************************
//
// Global storage probabilities that while in the current state we might see
// a particular state observation.
//
//*****************************************************************************
const float g_ppfEmitProb[GESTURE_NUM_STATES][GESTURE_NUM_STATES] =
{
    {0.897f, 0.01f, 0.01f, 0.01f, 0.1f,  0.0f,  0.0f},
    {0.01f,  0.99f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
    {0.01f,  0.0f,  0.99f, 0.0f,  0.0f,  0.0f,  0.0f},
    {0.01f,  0.0f,  0.0f,  0.99f, 0.0f,  0.0f,  0.0f},
    {0.01f,  0.0f,  0.0f,  0.0f,  0.99f, 0.01f, 0.01f},
    {0.0f,   0.0f,  0.0f,  0.0f,  0.01f, 0.99f, 0.0f},
    {0.0f,   0.0f,  0.0f,  0.0f,  0.01f, 0.0f, 0.99f}
};

//*****************************************************************************
//
// Global storage for deadbands.  These are guards against fast state to state
// transitions.  This defines the number of ticks during which the motion
// sensor is ignored.  For some states a quick transition is OK.  Other
// Other transitions can be followed by noise on the motion sensors.  This
// deadbanding helps filter out the noise after a state to state transition.
//
//*****************************************************************************
uint_fast16_t g_ui16DeadBands[GESTURE_NUM_STATES] = {15, 10, 10, 10, 15, 15,
                                                     15};

//*****************************************************************************
//
// Global storage for current GestureState
//
//*****************************************************************************
tGesture g_sGestureInst;

//*****************************************************************************
//
// Called by the NVIC as a result of I2C3 Interrupt. I2C3 is the I2C connection
// to the MPU9150.
//
//*****************************************************************************
void
MotionI2CIntHandler(void)
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
// Computes the product of a matrix and vector.
//
//*****************************************************************************
static void
MatrixVectorMul(float pfVectorOut[3], float ppfMatrixIn[3][3],
                float pfVectorIn[3])
{
    uint32_t ui32X, ui32Y;

    //
    // Loop through the rows of the matrix.
    //
    for(ui32Y = 0; ui32Y < 3; ui32Y++)
    {
        //
        // Initialize the value to zero
        //
        pfVectorOut[ui32Y] = 0;

        //
        // Loop through the columns of the matrix.
        //
        for(ui32X = 0; ui32X < 3; ui32X++)
        {
            //
            // The answer to this vector's row's value is the sum of each
            // column value multiplied by each vector's row value.
            //
            pfVectorOut[ui32Y] += (ppfMatrixIn[ui32Y][ui32X] *
                                   pfVectorIn[ui32X]);
        }
    }
}

//*****************************************************************************
//
// Attempt to classify current measurement into an event observation for
// purpose of feeding it into our modified Viterbi algorithm.
//
//*****************************************************************************
uint_fast16_t
GestureEmitClassify(tGesture* pGestInst, float* pfEulers, float* pfAccel,
                    float* pfAngVelocity)
{
    float fAccelMagnitude, fJerk;
    float pfAccelNet[3];
    float ppfDCM[3][3];

    //
    // Convert accel measurements to global coordinate frame then subtract off
    // gravity component. This leaves net acceleration from the user movement.
    //
    CompDCMMatrixGet(&g_sCompDCMInst, ppfDCM);
    MatrixVectorMul(pfAccelNet, ppfDCM, pfAccel);
    pfAccelNet[2] -= 9.81f;

    //
    // Calculate magnitude of the total acceleration from user.
    //
    fAccelMagnitude = (pfAccelNet[0] * pfAccelNet[0]) +
                      (pfAccelNet[1] * pfAccelNet[1]) +
                      (pfAccelNet[2] * pfAccelNet[2]);
    //
    // Subtract from previous accel magnitude.  This become delta acceleration
    // since we last called this function.  We call this jerk.
    //
    fJerk = fAccelMagnitude - g_fAccelMagnitudePrevious;
    g_fAccelMagnitudePrevious = fAccelMagnitude;

    //
    // Default to current state.  So the state transition and classification
    // switch state will only modify the observation if the state is
    // perceived to have changed.
    //
    if(pGestInst->ui16DeadBandCounter < g_ui16DeadBands[pGestInst->ui16State])
    {
        pGestInst->ui16DeadBandCounter++;
        return (pGestInst->ui16PrevState);
    }

    //
    // If state changed since our last time through the loop then reset our
    // deadband counter.
    //
    if(pGestInst->ui16PrevState != pGestInst->ui16State)
    {
        pGestInst->ui16DeadBandCounter = 0;
    }

    //
    // Save off the previous emit value before we change it.
    // Set the default current emit value to the current state.
    //
    pGestInst->ui16PrevEmit = pGestInst->ui16Emit;
    pGestInst->ui16Emit = pGestInst->ui16State;

    //
    // What observations we can make depend on what state we currently are in.
    //
    switch(pGestInst->ui16State)
    {
        //
        // The current state is IDLE
        //
        case GESTURE_STATE_IDLE:
        {
            //
            // order of if statement checks is important since some gestures
            // may exhibit more than one characteristic we want to prioritize
            // which ones get detected
            //
            if(pfAccelNet[2] > GESTURE_EMIT_THRESHOLD_ACCEL_UP)
            {
                //
                // Acceleration indicates board is going up.  report lifted
                // observation.
                //
                pGestInst->ui16Emit = GESTURE_STATE_LIFTED;
            }
            else if(fabs(pfAngVelocity[2]) >
                    GESTURE_EMIT_THRESHOLD_YAW_SCROLLING)
            {
                //
                // Yaw rate shows a spin of the board. This is a scroll
                //
                pGestInst->ui16Emit = GESTURE_STATE_SCROLLING;
                if(pfAngVelocity[0] < 0.0f)
                {
                    pGestInst->i8Direction = -1;
                }
                else
                {
                    pGestInst->i8Direction = 1;
                }
            }
            else if(fabs(pfAccelNet[0]) >
                    GESTURE_EMIT_THRESHOLD_ACCEL_ZOOMING)
            {
                //
                // Acceleration indicates sliding.  This is a "zoom"
                //
                pGestInst->ui16Emit = GESTURE_STATE_ZOOMING;
                if(pfAccelNet[0] < 0.0f)
                {
                    pGestInst->i8Direction = -1;
                }
                else
                {
                    pGestInst->i8Direction = 1;
                }
            }
            else if(((fabs(pfEulers[0]) > GESTURE_EMIT_THRESHOLD_MOUSING) ||
                    (fabs(pfEulers[1]) > GESTURE_EMIT_THRESHOLD_MOUSING)) &&
                    (fJerk < GESTURE_EMIT_THRESHOLD_ACCEL_MOUSING))
            {
                //
                // Either roll or pitch indicates we might be moving the cursor
                // since external accelerations can throw off the DCM output
                // ignore mouse movements when accelerating fast.
                //
                pGestInst->ui16Emit = GESTURE_STATE_MOUSING;
            }
            break;
        }

        //
        // Current state is moving the mouse. Characterized by roll and pitch
        // changes from the idle flat and level state.
        //
        case GESTURE_STATE_MOUSING:
        {
            //
            // If the roll and pitch drop below the threshold we are back to
            // idle.  otherwise stay in mouse mode.
            //
            if((fabs(pfEulers[0]) < GESTURE_EMIT_THRESHOLD_MOUSING) &&
               (fabs(pfEulers[1]) < GESTURE_EMIT_THRESHOLD_MOUSING))
            {
                pGestInst->ui16Emit = GESTURE_STATE_IDLE;
            }
            break;
        }

        //
        // Current State is lifted. Characterized by a quick jerk up of the
        // mouse into the air.
        //
        case GESTURE_STATE_LIFTED:
        {
            if(pfAccelNet[2] < GESTURE_EMIT_THRESHOLD_ACCEL_DOWN)
            {
                //
                // Accelerometer indicates we saw a down event. Go back to
                // idle state.
                //
                pGestInst->ui16Emit = GESTURE_STATE_IDLE;
            }
            else if(pfAngVelocity[1] < GESTURE_EMIT_THRESHOLD_TWIST_RIGHT)
            {
                //
                // Angular velocities show a twist right
                //
                pGestInst->ui16Emit = GESTURE_STATE_TWIST_RIGHT;
            }
            else if(pfAngVelocity[1] > GESTURE_EMIT_THRESHOLD_TWIST_LEFT)
            {
                //
                // Angular velocities show a twist left
                //
                pGestInst->ui16Emit = GESTURE_STATE_TWIST_LEFT;
            }
            break;
        }

        //
        // We are currently in the scrolling state. Characterized by a sharp
        // rotation or spin of the mouse around the Z access.
        //
        case GESTURE_STATE_SCROLLING:
        {
            if(fabs(pfAngVelocity[2]) < GESTURE_EMIT_THRESHOLD_YAW_SCROLLING)
            {
                //
                // angular velocity is now below the scrolling threshold so
                // we sense we are idle.
                //
                pGestInst->ui16Emit = GESTURE_STATE_IDLE;
            }
            break;
        }

        //
        // We are in the up and twisted state. Characterized by a lift state
        // followed by a wrist twist to the left or right.  Twist rate not
        // critical, we are sensing on twist angle.
        //
        case GESTURE_STATE_TWIST_RIGHT:
        case GESTURE_STATE_TWIST_LEFT:
        {
            if(fabs(pfEulers[1]) < GESTURE_EMIT_THRESHOLD_TWIST_LEVEL)
            {
                //
                // Eulers show that we are back to near level so return to
                // plain lifted state.
                //
                pGestInst->ui16Emit = GESTURE_STATE_LIFTED;
            }
            break;
        }

        //
        // We are in the zooming state. Characterized by a sharp acceleration
        // in the left/right axis while in idle mode.
        //
        case GESTURE_STATE_ZOOMING:
        {
            if(fabs(pfAccelNet[1]) < GESTURE_EMIT_THRESHOLD_ACCEL_ZOOMING)
            {
                //
                // Accel indicates we have dropped back below the zoom
                // threshold therefore report we are observing idle.
                //
                pGestInst->ui16Emit = GESTURE_STATE_IDLE;
            }
            break;
        }
    }

    //
    // Return the value of our current emit observation.
    //
    return(pGestInst->ui16Emit);
}

//*****************************************************************************
//
// Initialize the probability matrices and the Gesture Instance structure.
//
//*****************************************************************************
uint_fast16_t
GestureInit(tGesture* pGestInst, const float* pfInitProb,
            float (*ppfPath)[GESTURE_NUM_STATES],
            const float (*ppfTransitionProb)[GESTURE_NUM_STATES],
            const float (*ppfEmitProb)[GESTURE_NUM_STATES],
            uint_fast16_t ui16PathLength,
            uint_fast16_t ui16NumStates, uint_fast16_t ui16ObsState)
{
    uint_fast16_t ui16Idx, ui16State;
    float fMax;

    //
    // initialize all of the pointers and coutners.
    //
    pGestInst->ui16PathLength = ui16PathLength;
    pGestInst->ui16NumStates = ui16NumStates;
    pGestInst->pfInitProb = pfInitProb;
    pGestInst->ui16DeadBandCounter = 0;
    pGestInst->ui16PathCounter = 0;
    pGestInst->ppfPath = ppfPath;
    pGestInst->ppfTransitionProb = ppfTransitionProb;
    pGestInst->ppfEmitProb = ppfEmitProb;

    ui16State = 0;
    fMax = 0.0f;

    //
    // Search for most probable state based on initial state probabilities and
    // and the initial observation.
    //
    for(ui16Idx = 0; ui16Idx < pGestInst->ui16NumStates; ui16Idx++)
    {
        pGestInst->ppfPath[0][ui16Idx] = g_pfInitProb[ui16Idx] *
                                         g_ppfEmitProb[ui16Idx][ui16ObsState];

        if(pGestInst->ppfPath[0][ui16Idx] > fMax)
        {
            fMax = pGestInst->ppfPath[0][ui16Idx];
            ui16State = ui16Idx;
        }
    }

    //
    // Record the first state and first observations.
    //
    pGestInst->ui16State = ui16State;
    pGestInst->ui16PrevState = ui16State;
    pGestInst->ui16Emit = ui16ObsState;
    pGestInst->ui16PrevEmit = ui16ObsState;
    pGestInst->fProbStateMax = fMax;
    return (ui16State);
}

//*****************************************************************************
//
// Make sure that after our math the Sum of the path probabilities is 1.0.
//
//*****************************************************************************
void
GestureNormalize(tGesture * pGestInst)
{
    uint_fast16_t ui16Idx;
    float fSum;

    fSum = 0.0f;

    //
    // Calculate the sum of all the probabilities.
    //
    for(ui16Idx = 0; ui16Idx < pGestInst->ui16NumStates; ui16Idx++)
    {
        fSum += pGestInst->ppfPath[pGestInst->ui16PathCounter][ui16Idx];
    }

    //
    // Divide by the sum so that now the sum of the probabilities will be 1.0.
    //
    for(ui16Idx = 0; ui16Idx < pGestInst->ui16NumStates; ui16Idx++)
    {
        pGestInst->ppfPath[pGestInst->ui16PathCounter][ui16Idx] /= fSum;
    }
}

//*****************************************************************************
//
// Calculate the best estimate of our current state based on observed state and
// state history.
//
//*****************************************************************************
uint_fast16_t
GestureUpdate(tGesture *pGestInst, uint_fast16_t ui16ObsState)
{
    float fProbMax, fProb;
    uint_fast16_t ui16StateMax, ui16PrevProbIndex, ui16ProbIndex;
    uint_fast16_t ui16Idx0, ui16Idx1;

    //
    // Get the previous state prob index and increment the current index.
    //
    ui16PrevProbIndex = pGestInst->ui16PathCounter;
    pGestInst->ui16PathCounter++;
    if(pGestInst->ui16PathCounter >= pGestInst->ui16PathLength)
    {
        pGestInst->ui16PathCounter = 0;
    }

    //
    // Local variable for path counter to simplify indexing later.
    //
    ui16ProbIndex = pGestInst->ui16PathCounter;

    //
    // record prev state
    //
    pGestInst->ui16PrevState = pGestInst->ui16State;

    //
    // reset the max probabilities
    //
    fProbMax = 0.0f;
    ui16StateMax = 0;

    for(ui16Idx0 = 0; ui16Idx0 < pGestInst->ui16NumStates; ui16Idx0++)
    {
        //
        // iterate through each possible state
        //
        for(ui16Idx1 = 0; ui16Idx1 < pGestInst->ui16NumStates; ui16Idx1++)
        {
            //
            // calculate this states probability
            //
            fProb = pGestInst->ppfPath[ui16PrevProbIndex][ui16Idx1] *
                    pGestInst->ppfTransitionProb[ui16Idx1][ui16Idx0] *
                    pGestInst->ppfEmitProb[ui16Idx0][ui16ObsState];

            //
            // if this state has highest probability so far record it as new
            // maximum
            //
            if(fProb > fProbMax)
            {
                fProbMax = fProb;
                ui16StateMax = ui16Idx1;
            }
        }

        //
        // Record max probability for this state into the path.
        //
        pGestInst->ppfPath[ui16ProbIndex][ui16Idx0] = fProbMax;
    }

    //
    // Normalize all my previous probabilities back to sum of 1.0f
    //
    GestureNormalize(pGestInst);

    //
    // Record the new max probability and most likely state after normalizing.
    //
    pGestInst->fProbStateMax = pGestInst->ppfPath[ui16ProbIndex][ui16StateMax];
    pGestInst->ui16State = ui16StateMax;

    return(ui16StateMax);
}

//*****************************************************************************
//
// MPU9150 Sensor callback function.  Called at the end of MPU9150 sensor
// driver transactions. This is called from I2C interrupt context.
//
//*****************************************************************************
void MotionCallback(void* pvCallbackData, uint_fast8_t ui8Status)
{
    //
    // If the transaction succeeded set the data flag to indicate to
    // application that this transaction is complete and data may be ready.
    //
    if(ui8Status == I2CM_STATUS_SUCCESS)
    {
        //
        // Set the motion event flag to show that we have completed the
        // i2c transfer
        //
        HWREGBITW(&g_ui32Events, MOTION_EVENT) = 1;

        //
        // Turn on the LED to show we are ready to process motion date
        //
        g_pui32RGBColors[RED] = 0xFFFF;
        RGBColorSet(g_pui32RGBColors);

        if(g_ui8MotionState == MOTION_STATE_RUN);
        {
            //
            // Get local copies of the raw motion sensor data.
            //
            MPU9150DataAccelGetFloat(&g_sMPU9150Inst, g_pfAccel, g_pfAccel + 1,
                                     g_pfAccel + 2);

            MPU9150DataGyroGetFloat(&g_sMPU9150Inst, g_pfGyro, g_pfGyro + 1,
                                    g_pfGyro + 2);

            MPU9150DataMagnetoGetFloat(&g_sMPU9150Inst, g_pfMag, g_pfMag + 1,
                                       g_pfMag + 2);

            //
            // Update the DCM. Do this in the ISR so that timing between the
            // calls is consistent and accurate.
            //
            CompDCMMagnetoUpdate(&g_sCompDCMInst, g_pfMag[0], g_pfMag[1],
                                 g_pfMag[2]);
            CompDCMAccelUpdate(&g_sCompDCMInst, g_pfAccel[0], g_pfAccel[1],
                               g_pfAccel[2]);
            CompDCMGyroUpdate(&g_sCompDCMInst, -g_pfGyro[0], -g_pfGyro[1],
                              -g_pfGyro[2]);
            CompDCMUpdate(&g_sCompDCMInst);
        }
    }
    else
    {
        //
        // An Error occurred in the I2C transaction.
        //
        HWREGBITW(&g_ui32Events, MOTION_ERROR_EVENT) = 1;
        g_ui8MotionState = MOTION_STATE_ERROR;
        g_ui32RGBMotionBlinkCounter = g_ui32SysTickCount;
    }

    //
    // Store the most recent status in case it was an error condition
    //
    g_vui8ErrorFlag = ui8Status;
}

//*****************************************************************************
//
// Called by the NVIC as a result of GPIO port B interrupt event. For this
// application GPIO port B pin 2 is the interrupt line for the MPU9150
//
//*****************************************************************************
void
IntGPIOb(void)
{
    uint32_t ui32Status;

    ui32Status = GPIOIntStatus(GPIO_PORTB_BASE, true);

    //
    // Clear all the pin interrupts that are set
    //
    GPIOIntClear(GPIO_PORTB_BASE, ui32Status);

    //
    // Check which GPIO caused the interrupt event.
    //
    if(ui32Status & GPIO_PIN_2)
    {
        //
        // The MPU9150 data ready pin was asserted so start an I2C transfer
        // to go get the latest data from the device.
        //
        MPU9150DataRead(&g_sMPU9150Inst, MotionCallback, &g_sMPU9150Inst);
    }
}

//*****************************************************************************
//
// MPU9150 Application error handler.
//
//*****************************************************************************
void
MotionErrorHandler(char * pcFilename, uint_fast32_t ui32Line)
{
    //
    // Set terminal color to red and print error status and locations
    //
    UARTprintf("\033[31;1m");
    UARTprintf("Error: %d, File: %s, Line: %d\nSee I2C status definitions in "
               "utils\\i2cm_drv.h\n", g_vui8ErrorFlag, pcFilename, ui32Line);

    //
    // Return terminal color to normal
    //
    UARTprintf("\033[0m");
}

//*****************************************************************************
//
// Function to wait for the MPU9150 transactions to complete.
//
//*****************************************************************************
void
MotionI2CWait(char* pcFilename, uint_fast32_t ui32Line)
{
    //
    // Put the processor to sleep while we wait for the I2C driver to
    // indicate that the transaction is complete.
    //
    while((HWREGBITW(&g_ui32Events, MOTION_EVENT) == 0) &&
          (g_vui8ErrorFlag == 0))
    {
        //
        // Do Nothing
        //
    }

    //
    // clear the event flag.
    //
    HWREGBITW(&g_ui32Events, MOTION_EVENT) = 0;

    //
    // If an error occurred call the error handler immediately.
    //
    if(g_vui8ErrorFlag)
    {
        MotionErrorHandler(pcFilename, ui32Line);
        g_vui8ErrorFlag = 0;
    }

    return;
}

//*****************************************************************************
//
// Initialize the I2C, MPU9150 and Gesture systems.
//
//*****************************************************************************
void
MotionInit(void)
{
    //
    // Enable port B used for motion interrupt.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);

    //
    // The I2C3 peripheral must be enabled before use.
    //
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C3);
    ROM_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);

    //
    // Configure the pin muxing for I2C3 functions on port D0 and D1.
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

    //
    // Configure and Enable the GPIO interrupt. Used for INT signal from the
    // MPU9150
    //
    ROM_GPIOPinTypeGPIOInput(GPIO_PORTB_BASE, GPIO_PIN_2);
    GPIOIntEnable(GPIO_PORTB_BASE, GPIO_PIN_2);
    ROM_GPIOIntTypeSet(GPIO_PORTB_BASE, GPIO_PIN_2, GPIO_FALLING_EDGE);
    ROM_IntEnable(INT_GPIOB);

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
    // Set the motion state to initializing.
    //
    g_ui8MotionState = MOTION_STATE_INIT;

    //
    // Initialize the MPU9150 Driver.
    //
    MPU9150Init(&g_sMPU9150Inst, &g_sI2CInst, MPU9150_I2C_ADDRESS,
                MotionCallback, &g_sMPU9150Inst);

    //
    // Wait for transaction to complete
    //
    MotionI2CWait(__FILE__, __LINE__);

    //
    // Write application specifice sensor configuration such as filter settings
    // and sensor range settings.
    //
    g_sMPU9150Inst.pui8Data[0] = MPU9150_CONFIG_DLPF_CFG_94_98;
    g_sMPU9150Inst.pui8Data[1] = MPU9150_GYRO_CONFIG_FS_SEL_250;
    g_sMPU9150Inst.pui8Data[2] = (MPU9150_ACCEL_CONFIG_ACCEL_HPF_5HZ |
                                  MPU9150_ACCEL_CONFIG_AFS_SEL_2G);
    MPU9150Write(&g_sMPU9150Inst, MPU9150_O_CONFIG, g_sMPU9150Inst.pui8Data, 3,
                 MotionCallback, &g_sMPU9150Inst);

    //
    // Wait for transaction to complete
    //
    MotionI2CWait(__FILE__, __LINE__);

    //
    // Configure the data ready interrupt pin output of the MPU9150.
    //
    g_sMPU9150Inst.pui8Data[0] = (MPU9150_INT_PIN_CFG_INT_LEVEL |
                                  MPU9150_INT_PIN_CFG_INT_RD_CLEAR |
                                  MPU9150_INT_PIN_CFG_LATCH_INT_EN);
    g_sMPU9150Inst.pui8Data[1] = MPU9150_INT_ENABLE_DATA_RDY_EN;
    MPU9150Write(&g_sMPU9150Inst, MPU9150_O_INT_PIN_CFG,
                 g_sMPU9150Inst.pui8Data, 2, MotionCallback, &g_sMPU9150Inst);

    //
    // Wait for transaction to complete
    //
    MotionI2CWait(__FILE__, __LINE__);

    //
    // Initialize the DCM system.
    //
    CompDCMInit(&g_sCompDCMInst, 1.0f / ((float) MOTION_SAMPLE_FREQ_HZ),
                DCM_ACCEL_WEIGHT, DCM_GYRO_WEIGHT, DCM_MAG_WEIGHT);

    //
    // Initialize the gesture instance and establish a initial state estimate.
    //
    GestureInit(&g_sGestureInst, g_pfInitProb, g_ppfPath, g_ppfTransitionProb,
                g_ppfEmitProb, GESTURE_PATH_LENGTH, GESTURE_NUM_STATES,
                GESTURE_STATE_IDLE);
}

//*****************************************************************************
//
// Main function to handler motion events that are triggered by the MPU9150
// data ready interrupt.
//
//*****************************************************************************
void
MotionMain(void)
{
    switch(g_ui8MotionState)
    {
        //
        // This is our initial data set from the MPU9150, start the DCM.
        //
        case MOTION_STATE_INIT:
        {
            //
            // Check the read data buffer of the MPU9150 to see if the
            // Magnetometer data is ready and present. This may not be the case
            // for the first few data captures.
            //
            if(g_sMPU9150Inst.pui8Data[14] & AK8975_ST1_DRDY)
            {
                //
                // Get local copy of Accel and Mag data to feed to the DCM
                // start.
                //
                MPU9150DataAccelGetFloat(&g_sMPU9150Inst, g_pfAccel,
                                         g_pfAccel + 1, g_pfAccel + 2);
                MPU9150DataMagnetoGetFloat(&g_sMPU9150Inst, g_pfMag,
                                          g_pfMag + 1, g_pfMag + 2);
                MPU9150DataGyroGetFloat(&g_sMPU9150Inst, g_pfGyro,
                                        g_pfGyro + 1, g_pfGyro + 2);

                //
                // Feed the initial measurements to the DCM and start it.
                // Due to the structure of our MotionMagCallback function,
                // the floating point magneto data is already in the local
                // data buffer.
                //
                CompDCMMagnetoUpdate(&g_sCompDCMInst, g_pfMag[0], g_pfMag[1],
                                     g_pfMag[2]);
                CompDCMAccelUpdate(&g_sCompDCMInst, g_pfAccel[0], g_pfAccel[1],
                                   g_pfAccel[2]);
                CompDCMStart(&g_sCompDCMInst);

                //
                // Proceed to the run state.
                //
                g_ui8MotionState = MOTION_STATE_RUN;
            }

            //
            // Turn off the LED to show we are done processing motion data.
            //
            g_pui32RGBColors[RED] = 0;
            RGBColorSet(g_pui32RGBColors);

            //
            // Finished
            //
            break;
        }

        //
        // DCM has been started and we are ready for normal operations.
        //
        case MOTION_STATE_RUN:
        {
            //
            // Get the latest Euler data from the DCM. DCMUpdate is done
            // inside the interrupt routine to insure it is not skipped and
            // that the timing is consistent.
            //
            CompDCMComputeEulers(&g_sCompDCMInst, g_pfEulers,
                                 g_pfEulers + 1, g_pfEulers + 2);

            //
            // Pass the latest sensor data back to the Gesture system for
            // classification.  What state do i think i am in?
            //
            GestureEmitClassify(&g_sGestureInst, g_pfEulers, g_pfAccel, g_pfGyro);

            //
            // Update best guess state based on past history and current
            // estimate.
            //
            GestureUpdate(&g_sGestureInst, g_sGestureInst.ui16Emit);

            //
            // Turn off the LED to show we are done processing motion data.
            //
            g_pui32RGBColors[RED] = 0;
            RGBColorSet(g_pui32RGBColors);

            //
            // Finished
            //
            break;
        }

        //
        // An I2C error has occurred at some point. Usually these are due to
        // asynchronous resets of the main MCU and the I2C peripherals. This
        // can cause the slave to hold the bus and the MCU to think it cannot
        // send.  In practice there are ways to clear this condition.  They are
        // not implemented here.  To clear power cycle the board.
        //
        case MOTION_STATE_ERROR:
        {
            //
            // Our tick counter and blink mechanism may not be safe across
            // rollovers of the g_ui32SysTickCount variable.  This rollover
            // only occurs after 1.3+ years of continuous operation.
            //
            if(g_ui32SysTickCount > (g_ui32RGBMotionBlinkCounter + 20))
            {
                //
                // 20 ticks have expired since we last toggled so turn off the
                // LED and reset the counter.
                //
                g_ui32RGBMotionBlinkCounter = g_ui32SysTickCount;
                g_pui32RGBColors[RED] = 0;
                RGBColorSet(g_pui32RGBColors);
            }
            else if(g_ui32SysTickCount == (g_ui32RGBMotionBlinkCounter + 10))
            {
                //
                // 10 ticks have expired since the last counter reset.  turn
                // on the RED LED.
                //
                g_pui32RGBColors[RED] = 0xFFFF;
                RGBColorSet(g_pui32RGBColors);
            }
            break;
        }
    }
}

//*****************************************************************************
//
// Application specific mapping of the motion system states to a mouse packet.
//
// i8DeltaX and i8DeltaY are the change in cursor position for this report.
//
// ui8Buttons is the button bit masked state.  bit 0 is left button, bit 1 is
// right button.
//
//*****************************************************************************
bool
MotionMouseGet(int8_t *i8DeltaX, int8_t *i8DeltaY, uint8_t *ui8Buttons)
{
    //
    // Assume zero mouse movement by default
    //
    *i8DeltaX = 0;
    *i8DeltaY = 0;

    if(g_sGestureInst.ui16State == GESTURE_STATE_MOUSING)
    {
        //
        // Set DeltaX and DeltaY proportional to roll and pitch
        // Negate DeltaX to get the left right orientation correct.
        //
        *i8DeltaY = (int8_t)((g_pfEulers[0] / 0.087f));
        *i8DeltaX = (int8_t)((g_pfEulers[1] / -0.087f));

        return (true);
    }

    return (false);
}

//*****************************************************************************
//
// Application specific mapping of the motion system to keyboard packet.
//
// ui8Modifiers will contain the modifier keys such as ALT or SHIFT on return.
// ui8Keys must contain storage space allocated by caller for up to 6 keys.
//
// All keys and modifiers will be returned in the form of USB usage codes.
// See
//
//*****************************************************************************
bool
MotionKeyboardGet(uint8_t * ui8Modifiers, uint8_t *ui8Key, bool *bModifierHold,
                  bool *bKeyHold)
{
    *ui8Modifiers = 0;
    *ui8Key = 0;
    *bModifierHold = false;
    *bKeyHold = false;

    //
    // USB Is connected and not suspended so begin gesture checking
    //
    if(g_sGestureInst.ui16State == g_sGestureInst.ui16PrevState)
    {
        //
        // If this gesture state is the same as the previous gesture
        // state then we have nothing to do here.
        //
        return(false);
    }

    switch(g_sGestureInst.ui16State)
    {
        //
        // Mouse has experienced lift gesture
        //
        case GESTURE_STATE_LIFTED:
        {
            //
            // Always maintain the ALT regardless of our previous state.
            //
            *ui8Modifiers = HID_KEYB_LEFT_ALT;
            *bModifierHold = true;

            //
            // Check if previous state was idle. If so send TAB as well to
            // start the window selection session.
            //
            if(g_sGestureInst.ui16PrevState == GESTURE_STATE_IDLE)
            {
                *ui8Key = HID_KEYB_USAGE_TAB;
                *bKeyHold = false;
            }
            break;
        }

        //
        // Mouse experienced a sharp wrist twist to the left while
        // in the lifted state.
        //
        case GESTURE_STATE_TWIST_LEFT:
        {
            *ui8Modifiers = HID_KEYB_LEFT_ALT;
            *bModifierHold = true;
            *ui8Key = HID_KEYB_USAGE_LEFT_ARROW;
            *bKeyHold = false;

            break;
        }

        //
        // Mouse experienced a sharp twist to the right while in the
        // lifted state.
        //
        case GESTURE_STATE_TWIST_RIGHT:
        {
            *ui8Modifiers = HID_KEYB_LEFT_ALT;
            *bModifierHold = true;
            *ui8Key = HID_KEYB_USAGE_RIGHT_ARROW;
            *bKeyHold = false;

            break;
        }

        //
        // Mouse experienced a jerk forward or back while in the Idle state.
        //
        case GESTURE_STATE_ZOOMING:
        {
            if(g_sGestureInst.i8Direction > 0)
            {
                *ui8Modifiers = HID_KEYB_LEFT_CTRL;
                *bModifierHold = false;
                *ui8Key = HID_KEYB_USAGE_KEYPAD_PLUS;
                *bKeyHold = false;
            }
            else
            {
                *ui8Modifiers = HID_KEYB_LEFT_CTRL;
                *bModifierHold = false;
                *ui8Key = HID_KEYB_USAGE_KEYPAD_MINUS;
                *bKeyHold = false;
            }
            break;
        }

        //
        // Mouse experienced a sharp twist of the Z axis while in the
        // idle state.
        //
        case GESTURE_STATE_SCROLLING:
        {
            if(g_sGestureInst.i8Direction > 0)
            {
                *ui8Modifiers = 0;
                *bModifierHold = false;
                *ui8Key = HID_KEYB_USAGE_PAGE_UP;
                *bKeyHold = false;
            }
            else
            {
                *ui8Modifiers = 0;
                *bModifierHold = false;
                *ui8Key = HID_KEYB_USAGE_PAGE_DOWN;
                *bKeyHold = false;
            }
            break;
        }

        //
        // Mouse has gone back to the idle state from some other state.
        //
        case GESTURE_STATE_IDLE:
        {
            if(g_sGestureInst.ui16PrevState == GESTURE_STATE_LIFTED)
            {
                *ui8Modifiers = 0;
                *bModifierHold = false;
                *ui8Key = 0;
                *bKeyHold = false;
            }
            break;
        }
    }

    //
    // Indicate to caller that we had some data to be sent.
    //
    return(true);
}

