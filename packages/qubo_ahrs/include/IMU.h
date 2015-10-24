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

#include "IMUTypes.h"

typedef struct _IMUFrame
{
   char frameID;
   int payloadSize;
} IMUFrame;

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

class IMU
{
   public:
      IMU(std::string deviceFile);
      ~IMU();
      int openDevice();
      bool isOpen();
      void closeDevice();
      int loadConfig();
      int readIMUData();
   private:
      std::string _deviceFile;
      int _deviceFD;
      IMUData _rawData;
      int readRaw(char* blob, int bytes_to_read);
      int writeRaw(char* blob, int bytes);
      int writeCommand(IMUFrame frame, char* payload);
      int readDatagram();
      int readDataResp();
      int readConfigResp();
      int readTapsResp();

      /*Below is all hardcoded data from the protocol spec'd for the TRAX.
       *             | Command Name       | Packet Code | Payload Size
       ******************************************************************************/
      static IMUFrame kGetModInfo               = {0x01, EMPTY                      };
      static IMUFrame kGetModInfoResp           = {0x02, sizeof(ModInfo)            };
      static IMUFrame kSetDataComponents        = {0x03, IMU_RAW_N_FIELDS+1         };
      static IMUFrame kGetData                  = {0x04, EMPTY                      };
      static IMUFrame kGetDataResp              = {0x05, sizeof(RawData)            };
      static IMUFrame kSetConfigBoolean         = {0x06, sizeof(ConfigBoolean)      };
      static IMUFrame kSetConfigFloat32         = {0x06, sizeof(ConfigFloat32)      };
      static IMUFrame kSetConfigUInt8           = {0x06, sizeof(ConfigUInt8)        };
      static IMUFrame kSetConfigUInt32          = {0x06, sizeof(ConfigUInt32)       };
      static IMUFrame kGetConfig                = {0x07, sizeof(ConfigID)           };
      static IMUFrame kGetConfigResp            = {0x08, sizeof(ConfigID)           };
      static IMUFrame kSave                     = {0x09, EMPTY                      };
      static IMUFrame kStartCal                 = {0x0a, sizeof(CalOption)          };
      static IMUFrame kStopCal                  = {0x0b, EMPTY                      };
      static IMUFrame kSetFIRFiltersZero        = {0x0c, sizeof(FIRTaps_Zero)       };
      static IMUFrame kSetFIRFiltersFour        = {0x0c, sizeof(FIRTaps_Four)       };
      static IMUFrame kSetFIRFiltersEight       = {0x0c, sizeof(FIRTaps_Eight)      };
      static IMUFrame kSetFIRFiltersSixteen     = {0x0c, sizeof(FIRTaps_Sixteen)    };
      static IMUFrame kSetFIRFiltersThirtyTwo   = {0x0c, sizeof(FIRTaps_ThirtyTwo)  };
      static IMUFrame kGetFIRFilters            = {0x0d, sizeof(FIRFilter)          };
      static IMUFrame kGetFIRFiltersResp        = {0x0e, sizeof(FIRTaps_Zero)       };
      static IMUFrame kPowerDown                = {0x0f, EMPTY                      };
      static IMUFrame kSaveDone                 = {0x10, sizeof(SaveError)          };
      static IMUFrame kUserCalSampleCount       = {0x11, sizeof(SampleCount)        };
      static IMUFrame kUserCalScore             = {0x12, sizeof(UserCalScore)       };
      static IMUFrame kSetConfigDone            = {0x13, EMPTY                      };
      static IMUFrame kSetFIRFiltersDone        = {0x14, EMPTY                      };
      static IMUFrame kStartContinuousMode      = {0x15, EMPTY                      };
      static IMUFrame kStopContinousMode        = {0x16, EMPTY                      };
      static IMUFrame kPowerUpDone              = {0x17, EMPTY                      };
      static IMUFrame kSetAcqParams             = {0x18, sizeof(AcqParams)          };
      static IMUFrame kGetAcqParams             = {0x19, EMPTY                      };
      static IMUFrame kSetAcqParamsDone         = {0x1a, EMPTY                      };
      static IMUFrame kGetAcqParamsResp         = {0x1b, sizeof(AcqParams)          };
      static IMUFrame kPowerDownDone            = {0x1c, EMPTY                      };
      static IMUFrame kFactoryMagCoeff          = {0x1d, EMPTY                      };
      static IMUFrame kFactoryMagCoeffDone      = {0x1e, EMPTY                      };
      static IMUFrame kTakeUserCalSample        = {0x1f, EMPTY                      };
      static IMUFrame kFactoryAccelCoeff        = {0x24, EMPTY                      };
      static IMUFrame kFactoryAccelCoeffDone    = {0x25, EMPTY                      };
      static IMUFrame kSetFunctionalMode        = {0x4f, sizeof(FunctionalMode)     };
      static IMUFrame kGetFunctionalMode        = {0x50, EMPTY                      };
      static IMUFrame kGetFunctionalModeResp    = {0x51, sizeof(FunctionalMode)     };
      static IMUFrame kSetResetRef              = {0x6e, EMPTY                      };
      static IMUFrame kSetMagTruthMethod        = {0x77, sizeof(MagTruthMethod)     };
      static IMUFrame kGetMagTruthMethod        = {0x78, EMPTY                      };
      static IMUFrame kGetMagTruthMethodResp    = {0x79, sizeof(MagTruthMethod)     };
}
#endif
