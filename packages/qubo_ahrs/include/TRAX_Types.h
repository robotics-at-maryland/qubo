#ifndef TRAX_TYPES_H
#define TRAX_TYPES_H

/******************************************************************************
 * IMUTypes.h - IMU internal data types
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris1727@gmail.com>
 *
 * All rights reserved.
 ******************************************************************************/

// uintX_t types and such
#include <stdint.h>

/**
 * All of the following types will be assembled as-is, with no padding. 
 * The paired statement at the bottom is pack(pop)
 */
#pragma pack(push,1)

/******************************************************************************
 * HARDCODED PAYLOAD TYPES:
 * These should be exactly out of the specification file.
 ******************************************************************************/

#define EMPTY 0

typedef struct _ModInfo {
   char type[4];
   uint8_t rev[4];
} ModInfo;

typedef struct _ConfigBoolean {
   config_id_t id;
   bool value;
} ConfigBoolean;

typedef struct _ConfigFloat32 {
   config_id_t id;
   float value;
} ConfigFloat32;

typedef struct _ConfigUInt8 {
   config_id_t id;
   uint8_t value;
} ConfigUInt8;

typedef struct _ConfigUInt32 {
   config_id_t id;
   uint32_t value;
} ConfigUInt32;

typedef uint8_t MagTruthMethod;

typedef uint16_t SaveError;

typedef struct _AcqParams {
   uint8_t aquisition_mode;
   uint8_t flush_filter;
   float pni_reserved;
   float sample_delay;
} AcqParams;

typedef uint32_t CalOption;

typedef uint32_t SampleCount;

typedef struct _UserCalScore {
   float mag_cal_score;
   float pni_reserved;
   float accel_cal_score;
   float distribution_error;
   float tilt_error;
   float tilt_range;
} UserCalScore;

typedef uint8_t FunctionalMode;

typedef struct _FIRFilter {
   uint8_t byte_1;
   uint8_t byte_2;
} FIRFilter;

typedef struct _FIRTaps_Zero {
   FIRFilter filter_id;
   uint8_t count;
} FIRTaps_Zero;

typedef struct _FIRTaps_Four {
   FIRFilter filter_id;
   uint8_t count;
   double taps[4];
} FIRTaps_Four;

typedef struct _FIRTaps_Eight {
   FIRFilter filter_id;
   uint8_t count;
   double taps[8];
} FIRTaps_Eight;

typedef struct _FIRTaps_Sixteen {
   FIRFilter filter_id;
   uint8_t count;
   double taps[16];
} FIRTaps_Sixteen;

typedef struct _FIRTaps_ThirtyTwo {
   FIRFilter filter_id;
   uint8_t count;
   double taps[32];
} FIRTaps_ThirtyTwo;

#pragma pack(pop)
#endif
