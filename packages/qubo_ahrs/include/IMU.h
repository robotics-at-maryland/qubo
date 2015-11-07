#ifndef IMU_H
#define IMU_H

/*
 * IMU.h
 * Header file for IMU API.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris172@gmail.com>
 * All rights reserved.
 * 
 * Adapted from earlier work of Steve Moskovchenko and Joseph Lisee (2007)
 */

// Standard lib includes.
#include <string>
#include <string.h>
#include <stdint.h>

// Unix includes
#include <termios.h>
#include <unistd.h>

// Error handling
#include <stdexcept>

#include "TRAXTypes.h"

/**
 * USER-DEFINED-TYPES:
 * These types can be manipulated as to change the behavior of data flow.
 * Integrity and book-keeping are paramount, as many data payloads need a certain
 * format in order to be interpreted correctly.
 */

/**
 * Struct of data types being sent/retrieved from the IMU.
 * Each field should be a data_id_t, with a count of IDs at the beginning.
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
#pragma pack(push,1)
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
#pragma pack(pop)

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

/**
 * IMU Configuration data structure.
 * Stores all data associated with the hardware configuration.
 */
typedef struct _IMUConfig
{
   // Struct configs
   AcqParams acqParams;
   FIRTaps_ThirtyTwo filters;
   // Simple configs
   MagTruthMethod magTruthMethod;
   FunctionalMode mode;
   // Primitive configs
   float declination;
   int userCalNumPoints;
   int magCoeffSet;
   int accelCoeffSet;
   uint8_t mountingRef;
   uint8_t baudRate;
   bool trueNorth;
   bool bigEndian;
   bool userCalAutoSampling;
   bool milOut;
   bool hprDuringCal;
} IMUConfig;

/**
 * Exception class for handling IO/Data integrity errors.
 */
class IMUException : public std::runtime_error
{
   public:
      IMUException(std::string message)
         : runtime_error(message) {}
};

/**
 * IMU API class
 * Contains all the low-level I/O abstraction, and allows the user to communicate
 * with the IMU hardware at a high level. Connects to a unix serial/usb terminal
 * and speaks the PNI Binary protocol to send and recieve packets of data.
 */
class IMU
{
   public:
      // Front facing API function calls.
      /**
       * Constructor for a new IMU interface.
       * @param (std::string) unix device name
       */
      IMU(std::string deviceFile);

      /** Destructor that cleans up and closes the device. */
      ~IMU();

      /** 
       * Opens the device and configures the I/O terminal. 
       */
      void openDevice();

      /** 
       * Checks if the IMU is currently open and avaiable 
       * @return (bool) whether the IMU is avaliable for other API operations.
       */
      bool isOpen();

      /** Disconnectes from the device and closes the teriminal. */
      void closeDevice();

      /** 
       * Reads the hardware info from the IMU unit. 
       * @return (std::string) Hardware info
       */
      std::string getInfo();

      /** 
       * Reads the current configuration from the IMU.
       * @return an IMUConfig struct with the live config data inside.
       */
      IMUConfig readConfig();

      /** Sends the staged config to the hardware. */
      void sendConfig();

      /** Send the data poll format to the IMU. */
      void sendIMUDataFormat();

      /** 
       * Polls the IMU for position information.
       * @return an IMUData struct of formatted IMU data.
       */
      IMUData pollIMUData();
   private:
      // Internal functionality.
      /** Unix file name to connect to */
      std::string _deviceFile;
      /** Serial port for I/O with the AHRS */
      int _deviceFD;
      /** Data rate to communicate with */
      speed_t _termBaud;
      /** Storage for readings from the IMU for caching purposes. */
      IMUData _lastReading;
      /** Configuration for the IMU that is staged-to-send */
      IMUConfig _stagedConfig;
      /** Configuration read directly from the IMU. */
      IMUConfig _liveConfig;

      /** Low-level internal I/O functions */
      int readRaw(uint8_t* blob, uint16_t bytes_to_read);
      int writeRaw(uint8_t* blob, uint16_t bytes_to_write);

      /** Mid-level I/O functionality using the protocol definitions. */
      void writeCommand(Command cmd, const void* payload);
      void readCommand(Command cmd, void* target);
      void sendCommand(Command cmd, const void* payload, Command resp, void* target);
   private:
      /** Library of static protocol frame definitions to categorize frames. */
      static const Command kGetModInfo;
      static const Command kGetModInfoResp;
      static const Command kSetDataComponents;
      static const Command kGetData;
      static const Command kGetDataResp;
      static const Command kSetConfigBoolean;
      static const Command kSetConfigFloat32;
      static const Command kSetConfigUInt8;
      static const Command kSetConfigUInt32;
      static const Command kGetConfig;
      static const Command kGetConfigRespBoolean;
      static const Command kGetConfigRespFloat32;
      static const Command kGetConfigRespUInt8;
      static const Command kGetConfigRespUInt32;
      static const Command kSave;
      static const Command kStartCal;
      static const Command kStopCal;
      static const Command kSetFIRFiltersZero;
      static const Command kSetFIRFiltersFour;
      static const Command kSetFIRFiltersEight;
      static const Command kSetFIRFiltersSixteen;
      static const Command kSetFIRFiltersThirtyTwo;
      static const Command kGetFIRFilters;
      static const Command kGetFIRFiltersRespZero;
      static const Command kGetFIRFiltersRespFour;
      static const Command kGetFIRFiltersRespEight;
      static const Command kGetFIRFiltersRespSixteen;
      static const Command kGetFIRFiltersRespThirtyTwo;
      static const Command kPowerDown;
      static const Command kSaveDone;
      static const Command kUserCalSampleCount;
      static const Command kUserCalScore;
      static const Command kSetConfigDone;
      static const Command kSetFIRFiltersDone;
      static const Command kStartContinuousMode;
      static const Command kStopContinousMode;
      static const Command kPowerUpDone;
      static const Command kSetAcqParams;
      static const Command kGetAcqParams;
      static const Command kSetAcqParamsDone;
      static const Command kGetAcqParamsResp;
      static const Command kPowerDownDone;
      static const Command kFactoryMagCoeff;
      static const Command kFactoryMagCoeffDone;
      static const Command kTakeUserCalSample;
      static const Command kFactoryAccelCoeff;
      static const Command kFactoryAccelCoeffDone;
      static const Command kSetFunctionalMode;
      static const Command kGetFunctionalMode;
      static const Command kGetFunctionalModeResp;
      static const Command kSetResetRef;
      static const Command kSetMagTruthMethod;
      static const Command kGetMagTruthMethod;
      static const Command kGetMagTruthMethodResp;

      static const config_id_t kDeclination;
      static const config_id_t kTrueNorth;
      static const config_id_t kBigEndian;
      static const config_id_t kMountingRef;
      static const config_id_t kUserCalNumPoints;
      static const config_id_t kUserCalAutoSampling;
      static const config_id_t kBaudRate;
      static const config_id_t kMilOut;
      static const config_id_t kHPRDuringCal;
      static const config_id_t kMagCoeffSet;
      static const config_id_t kAccelCoeffSet;

      static const data_id_t kHeading;
      static const data_id_t kPitch;
      static const data_id_t kRoll;
      static const data_id_t kHeadingStatus;
      static const data_id_t kQuaternion;
      static const data_id_t kTemperature;
      static const data_id_t kDistortion;
      static const data_id_t kCalStatus;
      static const data_id_t kAccelX;
      static const data_id_t kAccelY;
      static const data_id_t kAccelZ;
      static const data_id_t kMagX;
      static const data_id_t kMagY;
      static const data_id_t kMagZ;
      static const data_id_t kGyroX;
      static const data_id_t kGyroY;
      static const data_id_t kGyroZ;

      static const IMUSpeed k2400;
      //static const IMUSpeed k3600;
      static const IMUSpeed k4800;
      //static const IMUSpeed k7200;
      static const IMUSpeed k9600;
      //static const IMUSpeed k14400;
      static const IMUSpeed k19200;
      //static const IMUSpeed k28800;
      static const IMUSpeed k38400;
      static const IMUSpeed k57600;
      static const IMUSpeed k115200;

      static const FIRFilter kFilterID;
      static const RawDataFields dataConfig;

};
#endif
