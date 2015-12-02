/******************************************************************************
 * IMU.cpp
 * IMU Device API implementation.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris172@gmail.com>
 * All rights reserved.
 * 
 * Adapted from earlier work of Steve Moskovchenko and Joseph Lisee (2007)
 ******************************************************************************/

// DEBUG includes
#ifdef DEBUG

// Debug printing
#include <stdio.h>

#endif

// UNIX Serial includes
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>

// Header include
#include "../include/IMU.h"

   IMU::IMU(std::string deviceFile, IMUSpeed speed) 
: _deviceFile(deviceFile), _termBaud(speed.baud), _deviceFD(-1), _timeout({1,0})
{ }

IMU::~IMU() { closeDevice(); }

void IMU::openDevice() {
   struct termios termcfg;
   int modemcfg = 0, fd = -1;
   /* Open the serial port and store into a file descriptor.
    * O_RDWR allows for bi-directional I/O, and
    * O_NONBLOCK makes it so that read/write does not block.
    */
   fd = open(_deviceFile.c_str(), O_RDWR, O_NONBLOCK);
   // Check to see if the device exists.
   if (fd == -1)
      throw IMUException("Device '"+_deviceFile+"' unavaliable.");
   // Read the config of the interface.
   if(tcgetattr(fd, &termcfg)) 
      throw IMUException("Unable to read terminal configuration.");

   // Set the baudrate for the terminal
   if(cfsetospeed(&termcfg, _termBaud))
      throw IMUException("Unable to set terminal output speed.");
   if(cfsetispeed(&termcfg, _termBaud))
      throw IMUException("Unable to set terminal intput speed.");

   // Set raw I/O rules to read and write the data purely.
   cfmakeraw(&termcfg);

   // Configure the read timeout (deciseconds)
   termcfg.c_cc[VTIME] = 0;
   // Configure the minimum number of chars for read.
   termcfg.c_cc[VMIN] = 1;

   // Push the configuration to the terminal NOW.
   if(tcsetattr(fd, TCSANOW, &termcfg))
      throw IMUException("Unable to set terminal configuration.");

   // Pull in the modem configuration
   if(ioctl(fd, TIOCMGET, &modemcfg))
      throw IMUException("Unable to read modem configuration.");
   // Enable Request to Send
   modemcfg |= TIOCM_RTS;
   // Push the modem config back to the modem.
   if(ioctl(fd, TIOCMSET, &modemcfg))
      throw IMUException("Unable to set modem configuration.");

   // Successful execution!
   _deviceFD = fd;
}

bool IMU::isOpen() {return _deviceFD >= 0;}

void IMU::assertOpen() { if (!isOpen()) throw IMUException("Device needs to be open!"); }

void IMU::closeDevice() {
   if (isOpen()) 
      close(_deviceFD);
   _deviceFD = -1;
}

std::string IMU::getInfo() {
   ModInfo info;
   assertOpen();
   sendCommand(kGetModInfo, NULL, kGetModInfoResp, &info);
   return std::string(((char*) &info), sizeof(ModInfo));
}

void IMU::setDeclination(float decl) {
   ConfigFloat32 data = {kDeclination, decl};
   assertOpen();
   sendCommand(kSetConfigFloat32, &data, kSetConfigDone, NULL);
   saveConfig();
}

float IMU::getDeclination() {
   ConfigFloat32 data;
   assertOpen();
   sendCommand(kGetConfig, &kDeclination, kGetConfigRespFloat32, &data);
   return data.value;
}

void IMU::setTrueNorth(bool north) {
   ConfigBoolean data = {kTrueNorth, north};
   assertOpen();
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getTrueNorth() {
   ConfigBoolean data;
   assertOpen();
   sendCommand(kGetConfig, &kTrueNorth, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setBigEndian(bool endian) {
   ConfigBoolean data = {kBigEndian, endian};
   assertOpen();
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getBigEndian() {
   ConfigBoolean data;
   assertOpen();
   sendCommand(kGetConfig, &kBigEndian, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setMounting(MountRef mount) {
   ConfigUInt8 data = {kMountingRef, mount};
   assertOpen();
   sendCommand(kSetConfigUInt8, &data, kSetConfigDone, NULL);
   saveConfig();
}

MountRef IMU::getMounting() {
   ConfigUInt8 data;
   assertOpen();
   sendCommand(kGetConfig, &kMountingRef, kGetConfigRespUInt8, &data);
   switch(data.id) {
      case STD_0:       return STD_0;
      case X_UP_0:      return X_UP_0;
      case Y_UP_0:      return Y_UP_0;
      case STD_90:      return STD_90;
      case STD_180:     return STD_180;
      case STD_270:     return STD_270;
      case Z_DOWN_0:    return Z_DOWN_0;
      case X_UP_90:     return X_UP_90;
      case X_UP_180:    return X_UP_180;
      case X_UP_270:    return X_UP_270;
      case Y_UP_90:     return Y_UP_90;
      case Y_UP_180:    return Y_UP_180;
      case Y_UP_270:    return Y_UP_270;
      case Z_DOWN_90:   return Z_DOWN_90;
      case Z_DOWN_180:  return Z_DOWN_180;
      case Z_DOWN_270:  return Z_DOWN_270;
      default:          break;
   }
   throw IMUException("Unknown mounting reference id reported.");
}

void IMU::setCalPoints(unsigned int points) {
   ConfigUInt32 data = {kUserCalNumPoints, points};
   assertOpen();
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

unsigned int IMU::getCalPoints() {
   ConfigUInt32 data;
   assertOpen();
   sendCommand(kGetConfig, &kUserCalNumPoints, kGetConfigRespUInt32, &data);
   return data.value;
}

void IMU::setAutoCalibration(bool cal) {
   ConfigBoolean data = {kUserCalAutoSampling, cal};
   assertOpen();
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getAutoCalibration() {
   ConfigBoolean data;
   assertOpen();
   sendCommand(kGetConfig, &kUserCalAutoSampling, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setBaudrate(IMUSpeed speed)
{
   ConfigUInt8 data = {kBaudRate, speed.id};
   assertOpen();
   sendCommand(kSetConfigUInt8, &data, kSetConfigDone, NULL);
   saveConfig();
}

IMUSpeed IMU::getBaudrate() {
   ConfigUInt8 data;
   assertOpen();
   sendCommand(kGetConfig, &kBaudRate, kGetConfigRespUInt8, &data);
   switch (data.value) {
      case k0.id:       return k0;
      case k2400.id:    return k2400;
      case k4800.id:    return k4800;
      case k9600.id:    return k9600;
      case k19200.id:   return k19200;
      case k38400.id:   return k38400;
      case k57600.id:   return k57600;
      case k115200.id:  return k115200;
      default:          break;
   }
   throw IMUException("Unknown device baudrate id reported.");
}

void IMU::setMils(bool mils) {
   ConfigBoolean data = {kMilOut, mils};
   assertOpen();
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getMils() {
   ConfigBoolean data;
   assertOpen();
   sendCommand(kGetConfig, &kMilOut, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setHPRCal(bool hpr) {
   ConfigBoolean data = {kHPRDuringCal, hpr};
   assertOpen();
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getHPRCal() {
   ConfigBoolean data;
   assertOpen();
   sendCommand(kGetConfig, &kHPRDuringCal, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setMagCalID(CalibrationID id) {
   ConfigUInt32 data = {kMagCoeffSet, id};
   assertOpen();
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

CalibrationID IMU::getMagCalID() {
   ConfigUInt32 data;
   assertOpen();
   sendCommand(kGetConfig, &kMagCoeffSet, kGetConfigRespUInt32, &data);
   switch(data.value) {
      case CAL_0: return CAL_0;
      case CAL_1: return CAL_1;
      case CAL_2: return CAL_2;
      case CAL_3: return CAL_3;
      case CAL_4: return CAL_4;
      case CAL_5: return CAL_5;
      case CAL_6: return CAL_6;
      case CAL_7: return CAL_7;
      default:    break;
   }
   throw IMUException("Unexpected device calibration id reported.");
}

void IMU::setAccelCalID(CalibrationID id) {
   ConfigUInt32 data = {kAccelCoeffSet, id};
   assertOpen();
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

CalibrationID IMU::getAccelCalID() {
   ConfigUInt32 data;
   assertOpen();
   sendCommand(kGetConfig, &kAccelCoeffSet, kGetConfigRespUInt32, &data);
   switch(data.value) {
      case CAL_0: return CAL_0;
      case CAL_1: return CAL_1;
      case CAL_2: return CAL_2;
      case CAL_3: return CAL_3;
      case CAL_4: return CAL_4;
      case CAL_5: return CAL_5;
      case CAL_6: return CAL_6;
      case CAL_7: return CAL_7;
      default:    break;
   }
   throw IMUException("Unexpected device calibration id reported.");
}

void IMU::setMagTruthMethod(TruthMethod method) {
   assertOpen();
   writeCommand(kSetMagTruthMethod, &method);
   saveConfig();
}

TruthMethod IMU::getMagTruthMethod() {
   MagTruthMethod data;
   assertOpen();
   sendCommand(kGetMagTruthMethod, NULL, kGetMagTruthMethodResp, &data);
   switch(data) {
      case STANDARD: return STANDARD;
      case TIGHT: return TIGHT;
      case AUTOMERGE: return AUTOMERGE;
      default:    break;
   }
   throw IMUException("Unexpected device truth method reported.");
}

void IMU::saveConfig() {
   SaveError err;
   assertOpen();
   sendCommand(kSave, NULL, kSaveDone, &err);
   if (err) 
      throw IMUException("Error while saving configuration to nonvolatile memory.");
}

void IMU::resetMagReference() {
   assertOpen();
   writeCommand(kSetResetRef, NULL);
}

void IMU::setAcqParams(AcqParams params) {
   sendCommand(kSetAcqParams, &params, kSetAcqParamsDone, NULL);
   saveConfig();
}

AcqParams IMU::getAcqParams() {
   AcqParams params;
   sendCommand(kGetAcqParams, NULL, kGetAcqParamsResp, &params);
   return params;
}

void IMU::sendIMUDataFormat()
{
   setBigEndian(false);
   writeCommand(kSetDataComponents, &dataConfig);
   saveConfig();
}

IMUData IMU::pollIMUData()
{
   RawData data;
   assertOpen();
   // Poll the IMU for a data message.
   sendCommand(kGetData, NULL, kGetDataResp, &data);
   // Copy all the data to the actual IMU storage.
   _lastReading.quaternion[0] = data.quaternion[0];
   _lastReading.quaternion[1] = data.quaternion[1];
   _lastReading.quaternion[2] = data.quaternion[2];
   _lastReading.quaternion[3] = data.quaternion[3];
   _lastReading.gyroX = data.gyroX;
   _lastReading.gyroY = data.gyroY;
   _lastReading.gyroZ = data.gyroZ;
   _lastReading.accelX = data.accelX;
   _lastReading.accelY = data.accelY;
   _lastReading.accelZ = data.accelZ;
   _lastReading.magX = data.magX;
   _lastReading.magY = data.magY;
   _lastReading.magZ = data.magZ;
   return _lastReading;
}

void IMU::startCalibration(CalType type) {
   assertOpen();
   writeCommand(kStartCal, &type);
}

void IMU::stopCalibration() {
   assertOpen();
   writeCommand(kStopCal, NULL);
}

int IMU::takeCalibrationPoint() {
   SampleCount point;
   assertOpen();
   sendCommand(kTakeUserCalSample, NULL, kUserCalSampleCount, &point);
   return point;
}

UserCalScore IMU::getCalibrationScore() {
   UserCalScore score;
   assertOpen();
   readCommand(kUserCalScore, &score);
   saveConfig();
   return score;
}

void IMU::resetMagCalibration() {
   assertOpen();
   sendCommand(kFactoryMagCoeff, NULL, kFactoryMagCoeffDone, NULL);
   saveConfig();
}

void IMU::resetAccelCalibration() {
   assertOpen();
   sendCommand(kFactoryAccelCoeff, NULL, kFactoryAccelCoeffDone, NULL);
   saveConfig();
}

void IMU::setAHRSMode(bool mode) {
   assertOpen();
   writeCommand(kSetFunctionalMode, &mode);
   saveConfig();
}

bool IMU::getAHRSMode() {
   bool mode;
   assertOpen();
   sendCommand(kGetFunctionalMode, NULL, kGetFunctionalModeResp, &mode);
   return mode;
}

void IMU::powerDown() {
   assertOpen();
   sendCommand(kPowerDown, NULL, kPowerDownDone, NULL);
}

void IMU::wakeUp() {
   bool mode;
   assertOpen();
   writeCommand(kGetFunctionalMode, NULL);
   readCommand(kPowerUpDone, NULL);
   readCommand(kGetFunctionalMode, &mode);
}

/** 
 * Reads raw data from the serial port.
 * @return number of bytes not read
 */
int IMU::readRaw(uint8_t* blob, uint16_t bytes_to_read)
{  
   // Keep track of the number of bytes read, and the number of fds that are ready.
   int bytes_read = 0, current_read = 0, fds_ready = 0;
   // Sets of file descriptors for use with select(2).
   fd_set read_fds, write_fds, except_fds;
   // Timeout in the form of {sec, usec}, for use with select(2).
   struct timeval timeout = _timeout;
   // Check if we need to read data at all.
   if (bytes_to_read > 0) {
      do {
         // Set up for the select call to handle timeouts.
         FD_ZERO(&read_fds);
         FD_ZERO(&write_fds);
         FD_ZERO(&except_fds);
         FD_SET(_deviceFD, &read_fds);
         // Wait until the device is ready to read, return the number of FDs avalaible.
         fds_ready = select(_deviceFD+1, &read_fds, &write_fds, &except_fds, &timeout);
         if (fds_ready == 1) {
            // The filedescriptor is ready to read.
            current_read = read(_deviceFD, (blob + bytes_read), 
                  (bytes_to_read - bytes_read));
            // If the read was successful, record the number of bytes read.
            if (current_read > 0) {
               bytes_read += current_read;
            }
         }
         // Keep reading until we've run out of data, or we couldnt read more data.
      } while ((bytes_read < bytes_to_read) && (fds_ready == 1) && (current_read > 0));
   }
   // Return the number of bytes we actually managed to read.
   return bytes_to_read - bytes_read;
}

/**
 * Write raw data to the serial port.
 * @return number of bytes not written.
 */
int IMU::writeRaw(uint8_t* blob, uint16_t bytes_to_write)
{
   // Keep track of the number of bytes written, and the number of fds that are ready.
   int bytes_written = 0, current_write = 0, fds_ready = 0;
   // Sets of file descriptors for use with select(2).
   fd_set read_fds, write_fds, except_fds;
   // Timeout in the form of {sec, usec}, for use with select(2).
   struct timeval timeout = _timeout;
   // Check if we need to write data at all.
   if (bytes_to_write > 0) {
      do {
         // Set up for the select call to handle timeouts.
         FD_ZERO(&read_fds);
         FD_ZERO(&write_fds);
         FD_ZERO(&except_fds);
         FD_SET(_deviceFD, &write_fds);
         // Wait until the device is ready to write, return the number of FDs avalaible.
         fds_ready = select(_deviceFD+1, &read_fds, &write_fds, &except_fds, &timeout);
         if (fds_ready == 1) {
            // The filedescriptor is ready to write.
            current_write = write(_deviceFD, (blob + bytes_written), 
                  (bytes_to_write - bytes_written));
            // If the write was successful, record the number of bytes written.
            if (current_write > 0) {
               bytes_written += current_write;
            }
         }
         // Keep writing until we've run out of data, or we couldnt write more data.
      } while ((bytes_written < bytes_to_write) && (fds_ready == 1) && (current_write > 0));
   }
   // Return the number of bytes we actually managed to write.
   return bytes_to_write - bytes_written;
}

/**
 * CRC update code conforming to the xmodem crc spec.
 */
uint16_t crc_xmodem_update (uint16_t crc, uint8_t data)
{
   int i;
   crc = crc ^ ((uint16_t)data << 8);
   for (i=0; i<8; i++)
   {
      if (crc & 0x8000)
         crc = (crc << 1) ^ 0x1021;
      else
         crc <<= 1;
   }
   return crc;
}

/**
 * CRC checksum code conforming to the xmodem crc spec.
 */
checksum_t crc16(uint8_t* data, bytecount_t bytes){
   uint16_t crc;
   for (crc = 0x0; bytes > 0; bytes--, data++){
      crc = crc_xmodem_update(crc, *data);
   }
   return (checksum_t) crc;
}

/**
 * Sends a command to the device and waits for the response.
 */
void IMU::sendCommand(Command send, const void* payload, Command resp, void* target)
{
   if ((target != NULL) && memset(target, 0, resp.payload_size) == NULL)
      throw IMUException("Unable to clear command response target memory.");
   writeCommand(send, payload);
   readCommand(resp, target);
}

#define BUF_SIZE 4096
#define ENDIAN16(A) ((A << 8) | (A >> 8))

void IMU::writeCommand(Command cmd, const void* payload)
{
#ifdef DEBUG
   printf("Writing command %s.\n",cmd.name);
#endif
   // Temporary storage to assemble the data packet.
   uint8_t datagram[BUF_SIZE];

   // Pointers to specific parts of the datagram.
   bytecount_t* bytecount = (bytecount_t*) datagram;
   uint8_t* frame = datagram + sizeof(bytecount_t);
   uint8_t* data = frame + sizeof(frameid_t);
   checksum_t* checksum = (checksum_t*) (data + cmd.payload_size);

   // Various sizes of the parts of the datagram.
   bytecount_t data_offset = sizeof(bytecount_t) + sizeof(frameid_t);
   bytecount_t checksum_size = data_offset + cmd.payload_size;
   bytecount_t total_size = checksum_size + sizeof(checksum_t);

   // Copy the total datagram size to the datagram. (big-endian conversion)
   *bytecount = ENDIAN16(total_size);
   // Copy the frameid from the given Command into the datagram.
   *frame = cmd.id;
   // Copy the payload to the datagram.
   memcpy(data, payload, cmd.payload_size);
   // Compute the checksum
   *checksum = crc16(datagram, checksum_size);
   // Convert from little-endian to big-endian
   *checksum = ENDIAN16(*checksum);

   // Attempt to write the datagram to the serial port.
   if (writeRaw(datagram, total_size))
      throw IMUException("Unable to write command.");
}

void IMU::readCommand(Command cmd, void* target)
{
#ifdef DEBUG
   printf("Reading command %s.\n", cmd.name);
#endif
   // Temporary storage for the data read from the datagram.
   uint8_t datagram[BUF_SIZE];
   // Pointer to the start of the frame
   uint8_t *frame = datagram + sizeof(bytecount_t);
   // Some storage for data pulled out of the datagram.
   bytecount_t total_size, checksum_size, frame_size;
   // Storage for the expected checksum.
   checksum_t checksum;

   // Read in the header of the datagram packet: UInt16.
   if (readRaw(datagram, sizeof(bytecount_t)))
      throw IMUException("Unable to read bytecount of incoming packet.");
   // Calculate the number of bytes to read from the header.
   total_size = *((bytecount_t*) datagram);
   // Convert from big-endian to little-endian.
   total_size = ENDIAN16(total_size);
   // Do not include the checksum in the checksum
   checksum_size = total_size - sizeof(checksum_t);
   // Do not include the bytecount in the frame.
   frame_size = checksum_size - sizeof(bytecount_t);

   // Read in the actual data frame + checksum
   if (readRaw(frame, frame_size + sizeof(checksum_t)))
      throw IMUException("Unable to read body of incoming packet");
   // Pull out the sent checksum
   checksum = *((checksum_t*)(frame + frame_size));
   // Convert from big-endian to little-endian.
   checksum = ENDIAN16(checksum);

   // Validate the existing checksum
   if (crc16(datagram, checksum_size) != checksum)
      throw IMUException("Incoming packet checksum invalid.");
   //Identify that the message recieved matches what was expected.
   if (*frame != cmd.id)
      throw IMUException("Incoming packet frameid unexpected.");

   // Copy the data into the given buffer.
   memcpy(target, frame + sizeof(frameid_t), cmd.payload_size);
}

#undef BUF_SIZE
#undef ENDIAN16

/*****************************************************************************************************************
 * Below is all hardcoded data from the protocol spec'd for the TRAX.
 *               | Command Name                  | ID  | Payload Size             | Command String
 *****************************************************************************************************************/
const Command IMU::kGetModInfo                  = {0x01, EMPTY,                     "kGetModInfo"};
const Command IMU::kGetModInfoResp              = {0x02, sizeof(ModInfo),           "kGetModInfoResp"};
const Command IMU::kSetDataComponents           = {0x03, sizeof(RawDataFields),     "kSetDataComponents"};
const Command IMU::kGetData                     = {0x04, EMPTY,                     "kGetData"};
const Command IMU::kGetDataResp                 = {0x05, sizeof(RawData),           "kGetDataResp"};
const Command IMU::kSetConfigBoolean            = {0x06, sizeof(ConfigBoolean),     "kSetConfigBoolean"};
const Command IMU::kSetConfigFloat32            = {0x06, sizeof(ConfigFloat32),     "kSetConfigFloat32"};
const Command IMU::kSetConfigUInt8              = {0x06, sizeof(ConfigUInt8),       "kSetConfigUInt8"};
const Command IMU::kSetConfigUInt32             = {0x06, sizeof(ConfigUInt32),      "kSetConfigUInt32"};
const Command IMU::kGetConfig                   = {0x07, sizeof(config_id_t),       "kGetConfig"};
const Command IMU::kGetConfigRespBoolean        = {0x08, sizeof(ConfigBoolean),     "kGetConfigRespBoolean"};
const Command IMU::kGetConfigRespFloat32        = {0x08, sizeof(ConfigFloat32),     "kGetConfigRespFloat32"};
const Command IMU::kGetConfigRespUInt8          = {0x08, sizeof(ConfigUInt8),       "kGetConfigRespUInt8"};
const Command IMU::kGetConfigRespUInt32         = {0x08, sizeof(ConfigUInt32),      "kGetConfigRespUInt32"};
const Command IMU::kSave                        = {0x09, EMPTY,                     "kSave"};
const Command IMU::kStartCal                    = {0x0a, sizeof(CalOption),         "kStartCal"};
const Command IMU::kStopCal                     = {0x0b, EMPTY,                     "kStopCal"};
const Command IMU::kSetFIRFiltersZero           = {0x0c, sizeof(FIRTaps_Zero),      "kSetFIRFiltersZero"};
const Command IMU::kSetFIRFiltersFour           = {0x0c, sizeof(FIRTaps_Four),      "kSetFIRFiltersFour"};
const Command IMU::kSetFIRFiltersEight          = {0x0c, sizeof(FIRTaps_Eight),     "kSetFIRFiltersEight"};
const Command IMU::kSetFIRFiltersSixteen        = {0x0c, sizeof(FIRTaps_Sixteen),   "kSetFIRFiltersSixteen"};
const Command IMU::kSetFIRFiltersThirtyTwo      = {0x0c, sizeof(FIRTaps_ThirtyTwo), "kSetFIRFiltersThirtyTwo"};
const Command IMU::kGetFIRFilters               = {0x0d, sizeof(FIRFilter),         "kGetFIRFilters"};
const Command IMU::kGetFIRFiltersRespZero       = {0x0e, sizeof(FIRTaps_Zero),      "kGetFIRFiltersRespZero"};
const Command IMU::kGetFIRFiltersRespFour       = {0x0e, sizeof(FIRTaps_Four),      "kGetFIRFiltersRespFour"};
const Command IMU::kGetFIRFiltersRespEight      = {0x0e, sizeof(FIRTaps_Eight),     "kGetFIRFiltersRespEight"};
const Command IMU::kGetFIRFiltersRespSixteen    = {0x0e, sizeof(FIRTaps_Sixteen),   "kGetFIRFiltersRespSixteen"};
const Command IMU::kGetFIRFiltersRespThirtyTwo  = {0x0e, sizeof(FIRTaps_ThirtyTwo), "kGetFIRFiltersRespThirtyTwo"};
const Command IMU::kPowerDown                   = {0x0f, EMPTY,                     "kPowerDown"};
const Command IMU::kSaveDone                    = {0x10, sizeof(SaveError),         "kSaveDone"};
const Command IMU::kUserCalSampleCount          = {0x11, sizeof(SampleCount),       "kUserCalSampleCount"};
const Command IMU::kUserCalScore                = {0x12, sizeof(UserCalScore),      "kUserCalScore"};
const Command IMU::kSetConfigDone               = {0x13, EMPTY,                     "kSetConfigDone"};
const Command IMU::kSetFIRFiltersDone           = {0x14, EMPTY,                     "kSetFIRFiltersDone"};
const Command IMU::kStartContinuousMode         = {0x15, EMPTY,                     "kStartContinuousMode"};
const Command IMU::kStopContinousMode           = {0x16, EMPTY,                     "kStopContinousMode"};
const Command IMU::kPowerUpDone                 = {0x17, EMPTY,                     "kPowerUpDone"};
const Command IMU::kSetAcqParams                = {0x18, sizeof(AcqParams),         "kSetAcqParams"};
const Command IMU::kGetAcqParams                = {0x19, EMPTY,                     "kGetAcqParams"};
const Command IMU::kSetAcqParamsDone            = {0x1a, EMPTY,                     "kSetAcqParamsDone"};
const Command IMU::kGetAcqParamsResp            = {0x1b, sizeof(AcqParams),         "kGetAcqParamsResp"};
const Command IMU::kPowerDownDone               = {0x1c, EMPTY,                     "kPowerDownDone"};
const Command IMU::kFactoryMagCoeff             = {0x1d, EMPTY,                     "kFactoryMagCoeff"};
const Command IMU::kFactoryMagCoeffDone         = {0x1e, EMPTY,                     "kFactoryMagCoeffDone"};
const Command IMU::kTakeUserCalSample           = {0x1f, EMPTY,                     "kTakeUserCalSample"};
const Command IMU::kFactoryAccelCoeff           = {0x24, EMPTY,                     "kFactoryAccelCoeff"};
const Command IMU::kFactoryAccelCoeffDone       = {0x25, EMPTY,                     "kFactoryAccelCoeffDone"};
const Command IMU::kSetFunctionalMode           = {0x4f, sizeof(FunctionalMode),    "kSetFunctionalMode"};
const Command IMU::kGetFunctionalMode           = {0x50, EMPTY,                     "kGetFunctionalMode"};
const Command IMU::kGetFunctionalModeResp       = {0x51, sizeof(FunctionalMode),    "kGetFunctionalModeResp"};
const Command IMU::kSetResetRef                 = {0x6e, EMPTY,                     "kSetResetRef"};
const Command IMU::kSetMagTruthMethod           = {0x77, sizeof(MagTruthMethod),    "kSetMagTruthMethod"};
const Command IMU::kGetMagTruthMethod           = {0x78, EMPTY,                     "kGetMagTruthMethod"};
const Command IMU::kGetMagTruthMethodResp       = {0x79, sizeof(MagTruthMethod),    "kGetMagTruthMethodResp"};

const config_id_t IMU::kDeclination             = 1;
const config_id_t IMU::kTrueNorth               = 2;
const config_id_t IMU::kBigEndian               = 6;
const config_id_t IMU::kMountingRef             = 10;
const config_id_t IMU::kUserCalNumPoints        = 12;
const config_id_t IMU::kUserCalAutoSampling     = 13;
const config_id_t IMU::kBaudRate                = 14;
const config_id_t IMU::kMilOut                  = 15;
const config_id_t IMU::kHPRDuringCal            = 16;
const config_id_t IMU::kMagCoeffSet             = 18;
const config_id_t IMU::kAccelCoeffSet           = 19;

const data_id_t IMU::kPitch                     = 0x18;
const data_id_t IMU::kRoll                      = 0x19;
const data_id_t IMU::kHeadingStatus             = 0x4f;
const data_id_t IMU::kQuaternion                = 0x4d;
const data_id_t IMU::kTemperature               = 0x07;
const data_id_t IMU::kDistortion                = 0x08;
const data_id_t IMU::kCalStatus                 = 0x09;
const data_id_t IMU::kAccelX                    = 0x15;
const data_id_t IMU::kAccelY                    = 0x16;
const data_id_t IMU::kAccelZ                    = 0x17;
const data_id_t IMU::kMagX                      = 0x1b;
const data_id_t IMU::kMagY                      = 0x1c;
const data_id_t IMU::kMagZ                      = 0x1d;
const data_id_t IMU::kGyroX                     = 0x4a;
const data_id_t IMU::kGyroY                     = 0x4b;
const data_id_t IMU::kGyroZ                     = 0x4c;

const FIRFilter IMU::kFilterID                  = {3,1};
const RawDataFields IMU::dataConfig             = {
   10, kQuaternion, kGyroX, kGyroY, kGyroZ,
   kAccelX, kAccelY, kAccelZ, kMagX, kMagY, kMagZ};


















