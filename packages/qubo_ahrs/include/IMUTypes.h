#ifndef IMU_TYPES_H
#define IMU_TYPES_H

/*
 * IMUTypes.h - IMU internal data types
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris172@gmail.com>
 * All rights reserved.
 * 
 */

#include <stdint.h>

/**
 * All of the following types will be assembled as-is, with no padding. 
 * The paired statement at the bottom is pack(pop)
 */
#pragma pack(push,1)

/**
 * USER-DEFINED-TYPES:
 * These types can be manipulated as to change the behavior of data flow.
 * Integrity and book-keeping are paramount, as many data payloads need a certain
 * format in order to be interpreted correctly.
 */

/**
 * Struct of data being sent/retrieved from the IMU
 * VERY IMPORTANT: each major field must be one of the types defined
 * in the PNI AHRS spec, and must be preceded by a garbage char var.
 * This is in order to read in the data directly, and discard the data IDs.
 * Additionally, there is one garbage idCount at the beginning.
 * ALSO: make sure to update IMU_RAW_N_FIELDS to the number of major fields.
 */
#define RAW_DATA_N_FIELDS 10
typedef struct _RawData {
   char idCount;
   char qID;
   float[4] quaternion;
   char gxID;
   float gyroX;
   char gyID;
   float gyroY;
   char gzID;
   float gyroZ;
   char axID;
   float accelX;
   char ayID;
   float accelY;
   char azID;
   float accelZ;
   char mxID;
   float magX;
   char myID;
   float magY;
   char mzID;
   float magZ;
} RawData;

/**
 * NON-USER-DEFINED-TYPES:
 * Please dont change any of these unless the spec changes, or there is a bug.
 * These should be exactly out of the specification file.
 */

#define EMPTY 0

typedef struct _ModInfo {
   char[4] type;
   char[4] rev;
} ModInfo;

typedef uint8_t ConfigID;

typedef struct _ConfigBoolean {
   ConfigID id;
   bool value;
} ConfigBoolean;

typedef struct _ConfigFloat32 {
   ConfigID id;
   float value;
} ConfigFloat32;

typedef struct _ConfigUInt8 {
   ConfigID id;
   uint8_t value;
} ConfigUInt8;

typedef struct _ConfigUInt32 {
   ConfigID id;
   uint32_t value;
} ConfigUInt32;

typedef uint8_t MagTruthMethod;

typedef uint16_t SaveError;

typedef struct _AcqParams {
   uint8_t aqusition_mode;
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
   FIRTaps_Zero FIRTaps;
   double[4] taps;
} FIRTaps_Four;

typedef struct _FIRTaps_Eight {
   FIRTaps_Zero FIRTaps;
   double[8] taps;
} FIRTaps_Eight;

typedef struct _FIRTaps_Sixteen {
   FIRTaps_Zero FIRTaps;
   double[16] taps;
} FIRTaps_Sixteen;

typedef struct _FIRTaps_ThirtyTwo {
   FIRTaps_Zero FIRTaps;
   double[32] taps;
} FIRTaps_ThirtyTwo;




#pragma pack(pop)
#endif
