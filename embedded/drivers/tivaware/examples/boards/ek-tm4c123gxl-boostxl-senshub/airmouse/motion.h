//*****************************************************************************
//
// app_motion.h - Prototypes for applications motion sensor utilities
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

#ifndef __MOTION_H__
#define __MOTION_H__

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
// Define MPU9150 I2C Address.
//
//*****************************************************************************
#define MPU9150_I2C_ADDRESS     0x68

//*****************************************************************************
//
// Define MPU9150 data sampling frequency.
//
//*****************************************************************************
#define MOTION_SAMPLE_FREQ_HZ   50

//*****************************************************************************
//
// Weights the DCM should use for each sensor.  Must add to 1.0
//
//*****************************************************************************
#define DCM_MAG_WEIGHT          0.2f
#define DCM_GYRO_WEIGHT         0.6f
#define DCM_ACCEL_WEIGHT        0.2f

//*****************************************************************************
//
// Define the states of the motion state machine.
//
//*****************************************************************************
#define MOTION_STATE_INIT       0
#define MOTION_STATE_RUN        1
#define MOTION_STATE_ERROR      2

//*****************************************************************************
//
// Define the states of the motion state machine.
//
//*****************************************************************************
#define TO_DEG(a)               ((a) * 57.295779513082320876798154814105f)

//*****************************************************************************
//
// Define the Gesture constants.
//
//*****************************************************************************
#define GESTURE_PATH_LENGTH     256
#define GESTURE_NUM_STATES      7

//*****************************************************************************
//
// Define the Gesture states
//
//*****************************************************************************
#define GESTURE_STATE_IDLE      0
#define GESTURE_STATE_MOUSING   1
#define GESTURE_STATE_ZOOMING   2
#define GESTURE_STATE_SCROLLING 3
#define GESTURE_STATE_LIFTED    4
#define GESTURE_STATE_TWIST_LEFT    \
                                5
#define GESTURE_STATE_TWIST_RIGHT   \
                                6

//*****************************************************************************
//
// Define the Gesture emit classification thresholds. Units are radians,
// meters / second and radians / second as appropriate.
//
//*****************************************************************************
#define GESTURE_EMIT_THRESHOLD_MOUSING          \
                                0.087f
#define GESTURE_EMIT_THRESHOLD_ACCEL_MOUSING    \
                                0.5f
#define GESTURE_EMIT_THRESHOLD_ACCEL_UP         \
                                10.0f
#define GESTURE_EMIT_THRESHOLD_ACCEL_DOWN       \
                                -7.0f
#define GESTURE_EMIT_THRESHOLD_ACCEL_ZOOMING    \
                                2.5f
#define GESTURE_EMIT_THRESHOLD_TWIST_RIGHT      \
                                -2.57f
#define GESTURE_EMIT_THRESHOLD_TWIST_LEFT       \
                                1.57F
#define GESTURE_EMIT_THRESHOLD_YAW_SCROLLING    \
                                1.57f
#define GESTURE_EMIT_THRESHOLD_TWIST_LEVEL      \
                                0.17F

//*****************************************************************************
//
// Type for Gesture algorithm states
//
//*****************************************************************************
typedef struct sGesture
{
    uint_fast16_t ui16State;
    uint_fast16_t ui16PrevState;
    int_fast8_t   i8Direction;
    uint_fast16_t ui16Emit;
    uint_fast16_t ui16PrevEmit;
    uint_fast16_t ui16DeadBandCounter;
    uint_fast16_t ui16PathCounter;
    uint_fast16_t ui16NumStates;
    uint_fast16_t ui16PathLength;
    float fProbStateMax;
    float (*ppfPath)[GESTURE_NUM_STATES];
    const float*  pfInitProb;
    const float (*ppfTransitionProb)[GESTURE_NUM_STATES];
    const float (*ppfEmitProb)[GESTURE_NUM_STATES];
}tGesture;

//*****************************************************************************
//
// Function Interface
//
//*****************************************************************************
extern void MotionCalStart(uint_fast8_t);
extern void MotionInit(void);
extern void MotionMain(void);
extern bool MotionMouseGet(int8_t *i8DeltaX, int8_t *i8DeltaY,
                           uint8_t *ui8Buttons);
extern bool MotionKeyboardGet(uint8_t *ui8Modifiers, uint8_t *ui8Key,
                              bool *bModifierHold, bool *bKeyHold);

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

#endif // __MOTION_H__
