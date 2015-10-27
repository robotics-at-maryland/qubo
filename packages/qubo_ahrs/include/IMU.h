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

// Unix includes (open, read, write, etc...)
#include <unistd.h>

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
 * in the PNI AHRS spec, and must be preceded by a garbage uint8_t var.
 * This is in order to read in the data directly, and discard the data IDs.
 * Additionally, there is one garbage idCount at the beginning.
 * ALSO: make sure to update IMU_RAW_N_FIELDS to the number of major fields.
 */
#pragma pack(push,1)
typedef struct _RawData {
   uint8_t idCount;
   data_id_t qID;
   float[4] quaternion;
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
   float[4] quaternion;
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

class IMU
{
   public:
      IMU(std::string deviceFile);
      ~IMU();
      int openDevice();
      bool isOpen();
      void closeDevice();
      int sendConfig();
      IMUConfig readConfig();
      IMUData readIMUData();
   private:
      std::string _deviceFile;
      int _deviceFD;
      IMUData _lastReading;
      IMUConfig _stagedConfig;
      IMUConfig _liveConfig;

      int readRaw(char* blob, int bytes_to_read);
      int writeRaw(char* blob, int bytes);
      int sendCommand(Command cmd, void* payload, Command resp, void* target);
      int writeCommand(Command cmd, void* payload);
      int readCommand(Command cmd, void* target);

      /**
       * Hardcoded data ids matching the above struct definitions.
       */
      static RawDataFields dataConfig = {10, 
         kQuaternion, 
         kGyroX, kGyroY, kGyroZ,
         kAccelX, kAccelY, kAccelZ,
         kmagX, magY, magZ};

      /******************************************************************************
       * Below is all hardcoded data from the protocol spec'd for the TRAX.
       *            | Command Name               | ID   | Payload Size
       ******************************************************************************/
      static Command kGetModInfo                = {0x01, EMPTY                      };
      static Command kGetModInfoResp            = {0x02, sizeof(ModInfo)            };
      static Command kSetDataComponents         = {0x03, sizeof(RawDataFields)      };
      static Command kGetData                   = {0x04, EMPTY                      };
      static Command kGetDataResp               = {0x05, sizeof(RawData)            };
      static Command kSetConfigBoolean          = {0x06, sizeof(ConfigBoolean)      };
      static Command kSetConfigFloat32          = {0x06, sizeof(ConfigFloat32)      };
      static Command kSetConfigUInt8            = {0x06, sizeof(ConfigUInt8)        };
      static Command kSetConfigUInt32           = {0x06, sizeof(ConfigUInt32)       };
      static Command kGetConfig                 = {0x07, sizeof(config_id_t)        };
      static Command kGetConfigRespBoolean      = {0x08, sizeof(ConfigBoolean)      };
      static Command kGetConfigRespFloat32      = {0x08, sizeof(ConfigFloat32)      };
      static Command kGetConfigRespUInt8        = {0x08, sizeof(ConfigUInt8)        };
      static Command kGetConfigRespUInt32       = {0x08, sizeof(ConfigUInt32)       };
      static Command kSave                      = {0x09, EMPTY                      };
      static Command kStartCal                  = {0x0a, sizeof(CalOption)          };
      static Command kStopCal                   = {0x0b, EMPTY                      };
      static Command kSetFIRFiltersZero         = {0x0c, sizeof(FIRTaps_Zero)       };
      static Command kSetFIRFiltersFour         = {0x0c, sizeof(FIRTaps_Four)       };
      static Command kSetFIRFiltersEight        = {0x0c, sizeof(FIRTaps_Eight)      };
      static Command kSetFIRFiltersSixteen      = {0x0c, sizeof(FIRTaps_Sixteen)    };
      static Command kSetFIRFiltersThirtyTwo    = {0x0c, sizeof(FIRTaps_ThirtyTwo)  };
      static Command kGetFIRFilters             = {0x0d, sizeof(FIRFilter)          };
      static Command kGetFIRFiltersRespZero     = {0x0e, sizeof(FIRTaps_Zero)       };
      static Command kGetFIRFiltersRespFour     = {0x0e, sizeof(FIRTaps_Four)       };
      static Command kGetFIRFiltersRespEight    = {0x0e, sizeof(FIRTaps_Eight)      };
      static Command kGetFIRFiltersRespSixteen  = {0x0e, sizeof(FIRTaps_Sixteen)    };
      static Command kGetFIRFiltersRespThirtyTwo= {0x0e, sizeof(FIRTaps_ThirtyTwo)  };
      static Command kPowerDown                 = {0x0f, EMPTY                      };
      static Command kSaveDone                  = {0x10, sizeof(SaveError)          };
      static Command kUserCalSampleCount        = {0x11, sizeof(SampleCount)        };
      static Command kUserCalScore              = {0x12, sizeof(UserCalScore)       };
      static Command kSetConfigDone             = {0x13, EMPTY                      };
      static Command kSetFIRFiltersDone         = {0x14, EMPTY                      };
      static Command kStartContinuousMode       = {0x15, EMPTY                      };
      static Command kStopContinousMode         = {0x16, EMPTY                      };
      static Command kPowerUpDone               = {0x17, EMPTY                      };
      static Command kSetAcqParams              = {0x18, sizeof(AcqParams)          };
      static Command kGetAcqParams              = {0x19, EMPTY                      };
      static Command kSetAcqParamsDone          = {0x1a, EMPTY                      };
      static Command kGetAcqParamsResp          = {0x1b, sizeof(AcqParams)          };
      static Command kPowerDownDone             = {0x1c, EMPTY                      };
      static Command kFactoryMagCoeff           = {0x1d, EMPTY                      };
      static Command kFactoryMagCoeffDone       = {0x1e, EMPTY                      };
      static Command kTakeUserCalSample         = {0x1f, EMPTY                      };
      static Command kFactoryAccelCoeff         = {0x24, EMPTY                      };
      static Command kFactoryAccelCoeffDone     = {0x25, EMPTY                      };
      static Command kSetFunctionalMode         = {0x4f, sizeof(FunctionalMode)     };
      static Command kGetFunctionalMode         = {0x50, EMPTY                      };
      static Command kGetFunctionalModeResp     = {0x51, sizeof(FunctionalMode)     };
      static Command kSetResetRef               = {0x6e, EMPTY                      };
      static Command kSetMagTruthMethod         = {0x77, sizeof(MagTruthMethod)     };
      static Command kGetMagTruthMethod         = {0x78, EMPTY                      };
      static Command kGetMagTruthMethodResp     = {0x79, sizeof(MagTruthMethod)     };

      static config_id_t kDeclination           = 1;
      static config_id_t kTrueNorth             = 2;
      static config_id_t kBigEndian             = 6;
      static config_id_t kMountingRef           = 10;
      static config_id_t kUserCalNumPoints      = 12;
      static config_id_t kUserCalAutoSampling   = 13;
      static config_id_t kBaudRate              = 14;
      static config_id_t kMilOut                = 15;
      static config_id_t kHPRDuringCal          = 16;
      static config_id_t kMagCoeffSet           = 18;
      static config_id_t kAccelCoeffSet         = 19;

      static data_id_t kHeading                 = 0x05;
      static data_id_t kPitch                   = 0x18;
      static data_id_t kRoll                    = 0x19;
      static data_id_t kHeadingStatus           = 0x4f;
      static data_id_t kQuaternion              = 0x4d;
      static data_id_t kTemperature             = 0x07;
      static data_id_t kDistortion              = 0x08;
      static data_id_t kCalStatus               = 0x09;
      static data_id_t kAccelX                  = 0x15;
      static data_id_t kAccelY                  = 0x16;
      static data_id_t kAccelZ                  = 0x17;
      static data_id_t kMagX                    = 0x1b;
      static data_id_t kMagY                    = 0x1c;
      static data_id_t kMagZ                    = 0x1d;
      static data_id_t kGyroX                   = 0x4a;
      static data_id_t kGyroY                   = 0x4b;
      static data_id_t kGyroZ                   = 0x4c;

      static FIRFilter kFilterID                = {3,1};

#endif
