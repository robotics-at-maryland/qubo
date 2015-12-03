#ifndef API_TYPES_H
#define API_TYPES_H

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
// speed_t type
#include <termios.h>
// std::smart_ptr
#include <memory>

/**
 * All of the following types will be assembled as-is, with no padding. 
 * The paired statement at the bottom is pack(pop)
 */
#pragma pack(push,1)

/******************************************************************************
 * PROTOCOL TYPES
 * Data types pertaining to low-level protocol types.
 ******************************************************************************/

typedef uint8_t frameid_t;
typedef uint16_t bytecount_t;
typedef uint16_t checksum_t;

typedef uint8_t config_id_t;

/** Specification of a command defined in the PNI binary protocol. */
typedef struct _Command
{
   frameid_t id;
   bytecount_t payload_size;
   const char* name;
} Command;

/** Specification for the baudrate of the serial link. */
typedef struct _IMUSpeed
{
   uint8_t id;
   speed_t baud;
} IMUSpeed;

/******************************************************************************
 * USER-DEFINED-TYPES:
 * These types can be manipulated as to change the behavior of data flow.
 * Integrity and book-keeping are paramount, as many data payloads need a certain
 * format in order to be interpreted correctly.
 ******************************************************************************/

/** Data ID that uniquely identifies each type of sensor data. */
typedef uint8_t data_id_t;

/**
 * Struct of data types being sent/retrieved from the IMU.
 * The idCount at the beginning is how many different ids are in the struct.
 */
typedef struct _RawDataFields {
   uint8_t idCount;
   data_id_t qID;
   data_id_t gxID;
   data_id_t gyID;
   data_id_t gzID;
   data_id_t axID;
   data_id_t ayID;
   data_id_t azID;
   data_id_t mxID;
   data_id_t myID;
   data_id_t mzID;
} RawDataFields;

/**
 * Struct of data being sent/retrieved from the IMU
 * VERY IMPORTANT: each major field must be one of the types defined
 * in the PNI AHRS spec, and must be preceded by a garbage data_id_t var.
 * This is in order to read in the data directly, and discard the data IDs.
 * Additionally, there is one garbage idCount at the beginning.
 * ALSO: make sure to update IMU_RAW_N_FIELDS to the number of major fields.
 */
typedef struct _RawData {
   uint8_t idCount;
   data_id_t qID;
   float quaternion[4];
   data_id_t gxID;
   float gyroX;
   data_id_t gyID;
   float gyroY;
   data_id_t gzID;
   float gyroZ;
   data_id_t axID;
   float accelX;
   data_id_t ayID;
   float accelY;
   data_id_t azID;
   float accelZ;
   data_id_t mxID;
   float magX;
   data_id_t myID;
   float magY;
   data_id_t mzID;
   float magZ;
} RawData;

/**
 * Data type storing formatted IMU data to be passed around
 */
typedef struct _IMUData
{
   float quaternion[4];
   float gyroX;
   float gyroY;
   float gyroZ;

   float accelX;
   float accelY;
   float accelZ;

   float magX;
   float magY;
   float magZ;
} IMUData;

/******************************************************************************
 * API STRUCT TYPES
 * These types are structs that abstract between 
 * the API and the underlying protocol.
 ******************************************************************************/

/*
 * Struct for configuring the data aquisitions.
 */
typedef struct _AcqConfig
{
   /** Delay between samples while in continous mode. */
   float sample_delay;
   /** true will flush the collected data every time a reading is taken. */
   bool flush_filter;
   /** true for polling aquisition, false for continuous */
   bool poll_mode;
} AcqConfig;

/*
 * Struct containing calibration report data.
 */
typedef struct _CalScore
{
/** Magnetometer score (smaller is better, <2 is best). */
  float mag_score;
  /** Accelerometer score (smaller is better, <1 is best) */
  float accel_score;
  /** Distribution quality (should be 0) */
  float dist_error;
  /** Tilt angle quality (should be 0) */
  float tilt_error;
  /** Range of tilt angle (larger is better) */
  float tilt_range;
} CalScore;

typedef enum _FilterCount
{
   F_0   = 0,
   F_4   = 4,
   F_8   = 8,
   F_16  = 16,
   F_32  = 32
} FilterCount;

typedef struct _FilterData
{
   FilterCount count;
   std::shared_ptr<double> coeffs;
} FilterData;
   

/******************************************************************************
 * ENUMERATED TYPES
 * Enums of fields that can only have one of a set of defined values.
 ******************************************************************************/

typedef enum _MountRef
{
   STD_0       = 1,
   X_UP_0      = 2,
   Y_UP_0      = 3,
   STD_90      = 4,
   STD_180     = 5,
   STD_270     = 6,
   Z_DOWN_0    = 7,
   X_UP_90     = 8,
   X_UP_180    = 9,
   X_UP_270    = 10,
   Y_UP_90     = 11,
   Y_UP_180    = 12,
   Y_UP_270    = 13,
   Z_DOWN_90   = 14,
   Z_DOWN_180  = 15,
   Z_DOWN_270  = 16
} MountRef;

typedef enum _CalibrationID
{
   CAL_0, CAL_1, CAL_2, CAL_3, CAL_4, CAL_5, CAL_6, CAL_7
} CalibrationID;

typedef enum _TruthMethod
{
   STANDARD    = 1,
   TIGHT       = 2,
   AUTOMERGE   = 3
} TruthMethod;

typedef enum _CalType
{
   FULL_RANGE = 10,
   TWO_DIM = 20,
   HARD_IRON_ONLY = 30,
   LIMITED_TILT = 40,
   ACCEL_ONLY = 100,
   MAG_AND_ACCEL = 110
} CalType;

#pragma pack(pop)
#endif
