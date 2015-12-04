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

// Unix includes
#include <unistd.h>

// Error handling
#include <stdexcept>

// uint*_t types
#include <stdint.h>
// shared_ptr type
#include <memory>
// speed_t type
#include <termios.h>

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
      /**
       * All of the following types will be assembled as-is, with no padding. 
       * The paired statement is pack(pop)
       */
#pragma pack(push,1)

      /************************************************************************
       * PRIMITIVE TYPEDEFS
       * Basic primitive types given more specific names.
       * Reflects the different data type sizes in low-level protocols.
       ************************************************************************/
      /** Identifier for commands, sent at the beginning of a frame. */
      typedef uint8_t frameid_t;
      /** Length of data sent in a single packet. */
      typedef uint16_t bytecount_t;
      /** Xmodem 16-bit CRC checksum sent at the end of a packet. */
      typedef uint16_t checksum_t;
      /** Identifier for configuration data */
      typedef uint8_t config_id_t;
      /** Identifier for sensor data */
      typedef uint8_t data_id_t;

   private:
      /************************************************************************
       * INTERNAL STRUCT TYPEDEFS
       * Structs used for internal organization and communication.
       ************************************************************************/

      /** Reference details about a command sent over serial. */
      typedef struct _Command {
         /** Frame id to be sent or recieved on the serial connection. */
         frameid_t id;
         /** Size of the frame expected for this command */
         bytecount_t payload_size;
         /** Name of the command for debugging. */
         const char* name;
      } Command;

      /******************************************************************************
       * HARDCODED PAYLOAD TYPES:
       * These should be exactly out of the specification file.
       ******************************************************************************/

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
   public:
      /************************************************************************
       * ENUMERATED TYPEDEFS
       * Enumerated types defining values for fields that can only take a
       * limited number of unique values.
       ************************************************************************/

      /** Mounting reference ID. */
      typedef enum _MountRef {
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

      /** Current calibration set ID. */
      typedef enum _CalibrationID {
         CAL_0, CAL_1, CAL_2, CAL_3, CAL_4, CAL_5, CAL_6, CAL_7
      } CalibrationID;

      /** Method for determining the truth of the magnetometer data. */
      typedef enum _TruthMethod {
         STANDARD    = 1,
         TIGHT       = 2,
         AUTOMERGE   = 3
      } TruthMethod;

      /** Type of calibration to perform. */
      typedef enum _CalType {
         FULL_RANGE = 10,
         TWO_DIM = 20,
         HARD_IRON_ONLY = 30,
         LIMITED_TILT = 40,
         ACCEL_ONLY = 100,
         MAG_AND_ACCEL = 110
      } CalType;

      /** FIR Filter coefficient counts. */
      typedef enum _FilterCount {
         F_0   = 0,
         F_4   = 4,
         F_8   = 8,
         F_16  = 16,
         F_32  = 32
      } FilterCount;

      /************************************************************************
       * API TYPEDEFS
       * Structs used to communicate with the user different information.
       ************************************************************************/

      /** Specification for the baudrate of the serial link. */
      typedef struct _IMUSpeed {
         /** Identifier for the device speed for the device. */
         uint8_t id;
         /** Identifier for the host speed for the computer. */
         speed_t baud;
      } IMUSpeed;

      /** Struct for configuring the data aquisitions. */
      typedef struct _AcqConfig {
         /** Delay between samples while in continous mode. */
         float sample_delay;
         /** true will flush the collected data every time a reading is taken. */
         bool flush_filter;
         /** true for polling aquisition, false for continuous */
         bool poll_mode;
      } AcqConfig;

      /** Struct containing calibration report data. */
      typedef struct _CalScore {
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

      /** FIR filter coefficient data with the filter count. */
      typedef struct _FilterData {
         FilterCount count;
         std::shared_ptr<double> coeffs;
      } FilterData;

      /******************************************************************************
       * USER DEFINED TYPEDEFS
       * These types can be manipulated as to change the behavior of data flow.
       * Integrity and book-keeping are paramount, as many data payloads need a certain
       * format in order to be interpreted correctly.
       ******************************************************************************/

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

      /** static const struct with the permanent data type config. */
      const RawDataFields dataConfig             = {
         10, kQuaternion, kGyroX, kGyroY, kGyroZ,
         kAccelX, kAccelY, kAccelZ, kMagX, kMagY, kMagZ};

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

   public:
      /** Data type storing formatted IMU data to be passed around. */
      typedef struct _IMUData {
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

#pragma pack(pop)
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
      /** Checksum functionality */
      uint16_t crc_xmodem_update (uint16_t crc, uint8_t data);
      checksum_t crc16(uint8_t* data, bytecount_t bytes);


      /** Mid-level I/O functionality using the protocol definitions. */
      Command inferCommand(Command hint, frameid_t id, bytecount_t size);
      Command readFrame(Command hint, void* blob);

      void writeCommand(Command cmd, const void* payload);
      void readCommand(Command cmd, void* target);
      void sendCommand(Command cmd, const void* payload, Command resp, void* target);
      void printCommand(Command cmd);
   private:
#define EMPTY 0
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
#undef EMPTY

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
};
#endif
