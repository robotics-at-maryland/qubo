#ifndef IMU_H
#define IMU_H

/*
 * IMU.h
 * Header file for IMU API.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris1727@gmail.com>
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

#include "API_Types.h"

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
   public: // Front-facing types that need to be exposed to the user.
#include "API_Types.h"
   private: // Backend protocol types.
#include "TRAX_Types.h"
   public: // Front facing API function calls.
      /**
       * Constructor for a new IMU interface.
       * @param (std::string) unix device name
       * @param (IMUSpeed) Baudrate to use for connection.
       */
      IMU(std::string deviceFile, IMUSpeed speed);

      /** Destructor that cleans up and closes the device. */
      ~IMU();

      /** 
       * Opens the device and configures the I/O terminal. 
       * Requires dialout permissions to the device in _deviceFile.
       */
      void openDevice();

      /** 
       * Checks if the IMU is currently open and avaiable 
       * @return (bool) whether the IMU is avaliable for other API operations.
       */
      bool isOpen();

      /** Ensures that the IMU is open, throws an exception otherwise. */
      void assertOpen();

      /** Disconnectes from the device and closes the teriminal. */
      void closeDevice();

      /** 
       * Reads the hardware info from the IMU unit. 
       * @return (std::string) Hardware info
       */
      std::string getInfo();

      /**
       * Set the declination angle used to determine true north.
       * This is not used if only magnetic north is needed.
       * This is an outside variable depending on global location.
       * @param (float32) -180 to +180 degrees.
       */
      void setDeclination(float decl);

      /**
       * Get the declination angle used to determine true north.
       * This is not used if only magnetic north is needed.
       * This is an outside variable depending on global location.
       * @return (float32) -180 to +180 degrees.
       */
      float getDeclination();

      /**
       * Set the true north flag.
       * If true north is enabled, accuracy depends on the declination.
       * Otherwise the magnetic north is used without correction.
       * @param (bool) true for true north, false for magnetic.
       */
      void setTrueNorth(bool north);

      /**
       * Get the true north flag.
       * If true north is enabled, accuracy depends on the declination.
       * Otherwise the magnetic north is used without correction.
       * @return (bool) true for true north, false for magnetic.
       */
      bool getTrueNorth();

      /**
       * Set the endian-ness of the data I/O.
       * Little-endian is used in linux/unix machines.
       * Big-endian is used in TRAX Studio/elsewhere.
       * @param (bool) true for big-endian, false for little-endian.
       */
      void setBigEndian(bool endian);

      /**
       * Get the endian-ness of the the data I/O.
       * Little-endian is used in linux/unix machines.
       * Big-endian is used in TRAX Studio/elsewhere.
       * @return (bool) true for big-endian, false for little-endian.
       */
      bool getBigEndian();

      /**
       * Set the mounting reference orientation.
       * @param (MountRef)
       */
      void setMounting(MountRef mount);

      /**
       * Get the mounting reference orientation.
       * @return (MountRef)
       */
      MountRef getMounting();

      /**
       * Set the number of points in a user calibration.
       * @param (int) Number of points to expect during calibration.
       */
      void setCalPoints(unsigned int points);

      /**
       * Get the number of points in a user calibration.
       * @return (unsigned int) Number of points to expect during calibration.
       */
      unsigned int getCalPoints();

      /**
       * Set the auto-calibration flag.
       * Auto-calibration automatically acquires a point if it is suitable.
       * Manual calibration allows the user to choose each exact point.
       * @param (bool) true for auto, false for manual.
       */
      void setAutoCalibration(bool cal);

      /**
       * Get the auto-calibration flag.
       * Auto-calibration automatically acquires a point if it is suitable.
       * Manual calibration allows the user to choose each exact point.
       * @return (bool) true for auto, false for manual.
       */
      bool getAutoCalibration();

      /** 
       * Set the baudrate to the device.
       * NOTE: change requires a full power-cycle to update.
       * @param (IMUSpeed) Speed to use on the connection.
       */
      void setBaudrate(IMUSpeed speed);

      /** 
       * Get the baudrate to the device.
       * @return (IMUSpeed) Speed to use on the connection.
       */
      IMUSpeed getBaudrate();

      /**
       * Set the Mils/Degree flag.
       * 6400 Mils/circle
       * 360 Degrees/circle
       * @param (bool) true for mils, false for degrees.
       */
      void setMils(bool mils);

      /**
       * Get the Mils/Degree flag.
       * 6400 Mils/circle
       * 360 Degrees/circle
       * @return (bool) true for mils, false for degrees.
       */
      bool getMils();

      /**
       * Set Heading/Pitch/Roll calibration flag.
       * If true, heading pitch and roll data is sent 
       * while the device is being calibrated.
       * @param (bool) true for output, false for no data.
       */
      void setHPRCal(bool hpr);

      /**
       * Get Heading/Pitch/Roll calibration flag.
       * If true, heading pitch and roll data is sent 
       * while the device is being calibrated.
       * @return (bool) true for output, false for no data.
       */
      bool getHPRCal();

      /**
       * Set the current magnetometer calibration id.
       * The device stores up to 8 different calibrations.
       * @param (CalibrationID)
       */
      void setMagCalID(CalibrationID id);

      /**
       * Get the current magnetometer calibration id.
       * The device stores up to 8 different calibrations.
       * @return (CalibrationID)
       */
      CalibrationID getMagCalID();

      /**
       * Set the current acclerometer calibration id.
       * The device stores up to 8 different calibrations.
       * @param (CalibrationID)
       */
      void setAccelCalID(CalibrationID id);

      /**
       * Get the current acclerometer calibration id.
       * The device stores up to 8 different calibrations.
       * @return (CalibrationID)
       */
      CalibrationID getAccelCalID();

      /**
       * Set the magnetic truth method.
       * @param (TruthMethod)
       */
      void setMagTruthMethod(TruthMethod method);

      /**
       * Get the magnetic truth method.
       * @param (TruthMethod)
       */
      TruthMethod getMagTruthMethod();

      /**
       * Saves the current configuration to non-volatile memory.
       * This persists across hard restarts and power loss.
       */
      void saveConfig();

      /**
       * Reset the magnetic field reference.
       * Aligns the HPR to the magnetometer/accel heading.
       * Use only when the device is stable and the magnetic field
       * is not changing significantly.
       */
      void resetMagReference();

      /**
       * Set Aquisition Configuration
       * @param (AcqParams)
       */
      void setAcqConfig(AcqConfig config);

      /**
       * Get Aquisition Parameters.
       * @return (AcqParams)
       */
      AcqConfig getAcqConfig();

      /** Send the data poll format to the IMU. */
      void sendIMUDataFormat();

      /** 
       * Polls the IMU for position information.
       * @return an IMUData struct of formatted IMU data.
       */
      IMUData pollIMUData();

      /**
       * Start User Calibration.
       * @param (CalType) type of calibration.
       */
      void startCalibration(CalType);

      /** Stop User Calibration. */
      void stopCalibration();

      /**
       * Take User Calibration Point.
       * @return (int) current cal point number.
       */
      int takeCalibrationPoint();

      /**
       * Get the calibration score for the calibration just completed.
       * @return (CalScore)
       */
      CalScore getCalibrationScore();

      /** Reset the magnetometer to factory calibration settings. */
      void resetMagCalibration();

      /** Reset the accelerometer to factory calibration settings. */
      void resetAccelCalibration();

      /**
       * Set the AHRS mode flag.
       * @param (bool) true for AHRS, false for Compass.
       */
      void setAHRSMode(bool ahrs);

      /**
       * Get the AHRS mode flag.
       * @param (bool) true for AHRS, false for Compass.
       */
      bool getAHRSMode();

      /**
       * Set the Finite-Impuse-Response filters coefficient array.
       * @param (FilterData) filter configuration
       */
      void setFIRFilters(FilterData filters);

      /**
       * Get the Finite-Impuse-Response filters coefficient array.
       * @return (FilterData) filter configuration
       */
      FilterData getFIRFilters();

      /** Power down the device to conserve power. */
      void powerDown();

      /** Wake up the device from sleep. */
      void wakeUp();

   private: // Internal functionality.
      /** Unix file name to connect to */
      std::string _deviceFile;
      /** Data rate to communicate with */
      speed_t _termBaud;
      /** Serial port for I/O with the AHRS */
      int _deviceFD;
      /** Timeout (sec,usec) on read/write */
      struct timeval _timeout;
      /** Storage for readings from the IMU for caching purposes. */
      IMUData _lastReading;

      /** Low-level internal I/O functions */
      int readRaw(void* blob, uint16_t bytes_to_read);
      int writeRaw(void* blob, uint16_t bytes_to_write);
      Command readFrame(Command hint, void* blob);

      /** Mid-level I/O functionality using the protocol definitions. */
      void writeCommand(Command cmd, const void* payload);
      void readCommand(Command cmd, void* target);
      void sendCommand(Command cmd, const void* payload, Command resp, void* target);
      void printCommand(Command cmd);
   private:
      /*****************************************************************************************************************
       * Below is all hardcoded data from the protocol spec'd for the TRAX.
       *                     | Command Name                  | ID  | Payload Size             | Command String
       *****************************************************************************************************************/
      static constexpr Command kGetModInfo                  = {0x01, EMPTY,                     "kGetModInfo"};
      static constexpr Command kGetModInfoResp              = {0x02, sizeof(ModInfo),           "kGetModInfoResp"};
      static constexpr Command kSetDataComponents           = {0x03, sizeof(RawDataFields),     "kSetDataComponents"};
      static constexpr Command kGetData                     = {0x04, EMPTY,                     "kGetData"};
      static constexpr Command kGetDataResp                 = {0x05, sizeof(RawData),           "kGetDataResp"};
      static constexpr Command kSetConfigBoolean            = {0x06, sizeof(ConfigBoolean),     "kSetConfigBoolean"};
      static constexpr Command kSetConfigFloat32            = {0x06, sizeof(ConfigFloat32),     "kSetConfigFloat32"};
      static constexpr Command kSetConfigUInt8              = {0x06, sizeof(ConfigUInt8),       "kSetConfigUInt8"};
      static constexpr Command kSetConfigUInt32             = {0x06, sizeof(ConfigUInt32),      "kSetConfigUInt32"};
      static constexpr Command kGetConfig                   = {0x07, sizeof(config_id_t),       "kGetConfig"};
      static constexpr Command kGetConfigRespBoolean        = {0x08, sizeof(ConfigBoolean),     "kGetConfigRespBoolean"};
      static constexpr Command kGetConfigRespFloat32        = {0x08, sizeof(ConfigFloat32),     "kGetConfigRespFloat32"};
      static constexpr Command kGetConfigRespUInt8          = {0x08, sizeof(ConfigUInt8),       "kGetConfigRespUInt8"};
      static constexpr Command kGetConfigRespUInt32         = {0x08, sizeof(ConfigUInt32),      "kGetConfigRespUInt32"};
      static constexpr Command kSave                        = {0x09, EMPTY,                     "kSave"};
      static constexpr Command kStartCal                    = {0x0a, sizeof(CalOption),         "kStartCal"};
      static constexpr Command kStopCal                     = {0x0b, EMPTY,                     "kStopCal"};
      static constexpr Command kSetFIRFiltersZero           = {0x0c, sizeof(FIRTaps_Zero),      "kSetFIRFiltersZero"};
      static constexpr Command kSetFIRFiltersFour           = {0x0c, sizeof(FIRTaps_Four),      "kSetFIRFiltersFour"};
      static constexpr Command kSetFIRFiltersEight          = {0x0c, sizeof(FIRTaps_Eight),     "kSetFIRFiltersEight"};
      static constexpr Command kSetFIRFiltersSixteen        = {0x0c, sizeof(FIRTaps_Sixteen),   "kSetFIRFiltersSixteen"};
      static constexpr Command kSetFIRFiltersThirtyTwo      = {0x0c, sizeof(FIRTaps_ThirtyTwo), "kSetFIRFiltersThirtyTwo"};
      static constexpr Command kGetFIRFilters               = {0x0d, sizeof(FIRFilter),         "kGetFIRFilters"};
      static constexpr Command kGetFIRFiltersRespZero       = {0x0e, sizeof(FIRTaps_Zero),      "kGetFIRFiltersRespZero"};
      static constexpr Command kGetFIRFiltersRespFour       = {0x0e, sizeof(FIRTaps_Four),      "kGetFIRFiltersRespFour"};
      static constexpr Command kGetFIRFiltersRespEight      = {0x0e, sizeof(FIRTaps_Eight),     "kGetFIRFiltersRespEight"};
      static constexpr Command kGetFIRFiltersRespSixteen    = {0x0e, sizeof(FIRTaps_Sixteen),   "kGetFIRFiltersRespSixteen"};
      static constexpr Command kGetFIRFiltersRespThirtyTwo  = {0x0e, sizeof(FIRTaps_ThirtyTwo), "kGetFIRFiltersRespThirtyTwo"};
      static constexpr Command kPowerDown                   = {0x0f, EMPTY,                     "kPowerDown"};
      static constexpr Command kSaveDone                    = {0x10, sizeof(SaveError),         "kSaveDone"};
      static constexpr Command kUserCalSampleCount          = {0x11, sizeof(SampleCount),       "kUserCalSampleCount"};
      static constexpr Command kUserCalScore                = {0x12, sizeof(UserCalScore),      "kUserCalScore"};
      static constexpr Command kSetConfigDone               = {0x13, EMPTY,                     "kSetConfigDone"};
      static constexpr Command kSetFIRFiltersDone           = {0x14, EMPTY,                     "kSetFIRFiltersDone"};
      static constexpr Command kStartContinuousMode         = {0x15, EMPTY,                     "kStartContinuousMode"};
      static constexpr Command kStopContinousMode           = {0x16, EMPTY,                     "kStopContinousMode"};
      static constexpr Command kPowerUpDone                 = {0x17, EMPTY,                     "kPowerUpDone"};
      static constexpr Command kSetAcqParams                = {0x18, sizeof(AcqParams),         "kSetAcqParams"};
      static constexpr Command kGetAcqParams                = {0x19, EMPTY,                     "kGetAcqParams"};
      static constexpr Command kSetAcqParamsDone            = {0x1a, EMPTY,                     "kSetAcqParamsDone"};
      static constexpr Command kGetAcqParamsResp            = {0x1b, sizeof(AcqParams),         "kGetAcqParamsResp"};
      static constexpr Command kPowerDownDone               = {0x1c, EMPTY,                     "kPowerDownDone"};
      static constexpr Command kFactoryMagCoeff             = {0x1d, EMPTY,                     "kFactoryMagCoeff"};
      static constexpr Command kFactoryMagCoeffDone         = {0x1e, EMPTY,                     "kFactoryMagCoeffDone"};
      static constexpr Command kTakeUserCalSample           = {0x1f, EMPTY,                     "kTakeUserCalSample"};
      static constexpr Command kFactoryAccelCoeff           = {0x24, EMPTY,                     "kFactoryAccelCoeff"};
      static constexpr Command kFactoryAccelCoeffDone       = {0x25, EMPTY,                     "kFactoryAccelCoeffDone"};
      static constexpr Command kSetFunctionalMode           = {0x4f, sizeof(FunctionalMode),    "kSetFunctionalMode"};
      static constexpr Command kGetFunctionalMode           = {0x50, EMPTY,                     "kGetFunctionalMode"};
      static constexpr Command kGetFunctionalModeResp       = {0x51, sizeof(FunctionalMode),    "kGetFunctionalModeResp"};
      static constexpr Command kSetResetRef                 = {0x6e, EMPTY,                     "kSetResetRef"};
      static constexpr Command kSetMagTruthMethod           = {0x77, sizeof(MagTruthMethod),    "kSetMagTruthMethod"};
      static constexpr Command kGetMagTruthMethod           = {0x78, EMPTY,                     "kGetMagTruthMethod"};
      static constexpr Command kGetMagTruthMethodResp       = {0x79, sizeof(MagTruthMethod),    "kGetMagTruthMethodResp"};

      static constexpr config_id_t kDeclination             = 1;
      static constexpr config_id_t kTrueNorth               = 2;
      static constexpr config_id_t kBigEndian               = 6;
      static constexpr config_id_t kMountingRef             = 10;
      static constexpr config_id_t kUserCalNumPoints        = 12;
      static constexpr config_id_t kUserCalAutoSampling     = 13;
      static constexpr config_id_t kBaudRate                = 14;
      static constexpr config_id_t kMilOut                  = 15;
      static constexpr config_id_t kHPRDuringCal            = 16;
      static constexpr config_id_t kMagCoeffSet             = 18;
      static constexpr config_id_t kAccelCoeffSet           = 19;

      static constexpr data_id_t kPitch                     = 0x18;
      static constexpr data_id_t kRoll                      = 0x19;
      static constexpr data_id_t kHeadingStatus             = 0x4f;
      static constexpr data_id_t kQuaternion                = 0x4d;
      static constexpr data_id_t kTemperature               = 0x07;
      static constexpr data_id_t kDistortion                = 0x08;
      static constexpr data_id_t kCalStatus                 = 0x09;
      static constexpr data_id_t kAccelX                    = 0x15;
      static constexpr data_id_t kAccelY                    = 0x16;
      static constexpr data_id_t kAccelZ                    = 0x17;
      static constexpr data_id_t kMagX                      = 0x1b;
      static constexpr data_id_t kMagY                      = 0x1c;
      static constexpr data_id_t kMagZ                      = 0x1d;
      static constexpr data_id_t kGyroX                     = 0x4a;
      static constexpr data_id_t kGyroY                     = 0x4b;
      static constexpr data_id_t kGyroZ                     = 0x4c;
   public:
      static constexpr IMUSpeed k0        = {0,    B0};
      static constexpr IMUSpeed k2400     = {4,    B2400};
      static constexpr IMUSpeed k4800     = {6,    B4800};
      static constexpr IMUSpeed k9600     = {8,    B9600};
      static constexpr IMUSpeed k19200    = {10,   B19200};
      static constexpr IMUSpeed k38400    = {12,   B38400};
      static constexpr IMUSpeed k57600    = {13,   B57600};
      static constexpr IMUSpeed k115200   = {14,   B115200};
   private:
      static const RawDataFields dataConfig;

};
#endif
