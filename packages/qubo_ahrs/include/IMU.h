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
#include "TRAX_Types.h"

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
       * Set Aquisition Parameters.
       * @param (AcqParams)
       */
      void setAcqParams(AcqParams params);

      /**
       * Get Aquisition Parameters.
       * @return (AcqParams)
       */
      AcqParams getAcqParams();

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
       * Get User Calibration Score.
       * @return (UserCalScore)
       */
      UserCalScore getCalibrationScore();

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
      /** Timeout (usec) on read/write */
      int _timeout;
      /** Storage for readings from the IMU for caching purposes. */
      IMUData _lastReading;

      /** Low-level internal I/O functions */
      int readRaw(uint8_t* blob, uint16_t bytes_to_read);
      int writeRaw(uint8_t* blob, uint16_t bytes_to_write);

      /** Mid-level I/O functionality using the protocol definitions. */
      void writeCommand(Command cmd, const void* payload);
      void readCommand(Command cmd, void* target);
      void sendCommand(Command cmd, const void* payload, Command resp, void* target);
      void printCommand(Command cmd);
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
      static const FIRFilter kFilterID;
      static const RawDataFields dataConfig;

};
#endif
