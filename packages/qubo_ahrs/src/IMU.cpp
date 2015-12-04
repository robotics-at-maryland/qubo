/******************************************************************************
 * IMU.cpp
 * IMU Device API implementation.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris1727@gmail.com>
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
   sendCommand(kGetModInfo, NULL, kGetModInfoResp, &info);
   return std::string(((char*) &info), sizeof(ModInfo));
}

void IMU::setDeclination(float decl) {
   ConfigFloat32 data = {kDeclination, decl};
   sendCommand(kSetConfigFloat32, &data, kSetConfigDone, NULL);
   saveConfig();
}

float IMU::getDeclination() {
   ConfigFloat32 data;
   sendCommand(kGetConfig, &kDeclination, kGetConfigRespFloat32, &data);
   return data.value;
}

void IMU::setTrueNorth(bool north) {
   ConfigBoolean data = {kTrueNorth, north};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getTrueNorth() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kTrueNorth, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setBigEndian(bool endian) {
   ConfigBoolean data = {kBigEndian, endian};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getBigEndian() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kBigEndian, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setMounting(MountRef mount) {
   ConfigUInt8 data = {kMountingRef, mount};
   sendCommand(kSetConfigUInt8, &data, kSetConfigDone, NULL);
   saveConfig();
}

MountRef IMU::getMounting() {
   ConfigUInt8 data;
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
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

unsigned int IMU::getCalPoints() {
   ConfigUInt32 data;
   sendCommand(kGetConfig, &kUserCalNumPoints, kGetConfigRespUInt32, &data);
   return data.value;
}

void IMU::setAutoCalibration(bool cal) {
   ConfigBoolean data = {kUserCalAutoSampling, cal};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getAutoCalibration() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kUserCalAutoSampling, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setBaudrate(IMUSpeed speed)
{
   ConfigUInt8 data = {kBaudRate, speed.id};
   sendCommand(kSetConfigUInt8, &data, kSetConfigDone, NULL);
   saveConfig();
}

IMUSpeed IMU::getBaudrate() {
   ConfigUInt8 data;
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
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getMils() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kMilOut, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setHPRCal(bool hpr) {
   ConfigBoolean data = {kHPRDuringCal, hpr};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool IMU::getHPRCal() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kHPRDuringCal, kGetConfigRespBoolean, &data);
   return data.value;
}

void IMU::setMagCalID(CalibrationID id) {
   ConfigUInt32 data = {kMagCoeffSet, id};
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

CalibrationID IMU::getMagCalID() {
   ConfigUInt32 data;
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
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

CalibrationID IMU::getAccelCalID() {
   ConfigUInt32 data;
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
   writeCommand(kSetMagTruthMethod, &method);
   saveConfig();
}

TruthMethod IMU::getMagTruthMethod() {
   MagTruthMethod data;
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
   sendCommand(kSave, NULL, kSaveDone, &err);
   if (err) 
      throw IMUException("Error while saving configuration to nonvolatile memory.");
}

void IMU::resetMagReference() {
   writeCommand(kSetResetRef, NULL);
}

void IMU::setAcqConfig(AcqConfig config) {
   AcqParams params = {config.poll_mode, config.flush_filter, 0, config.sample_delay};
   sendCommand(kSetAcqParams, &params, kSetAcqParamsDone, NULL);
   saveConfig();
}

AcqConfig IMU::getAcqConfig() {
   AcqParams params;
   sendCommand(kGetAcqParams, NULL, kGetAcqParamsResp, &params);
   return { params.sample_delay,
            ((params.flush_filter) ? true : false), 
            ((params.aquisition_mode) ? true : false)};
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
   writeCommand(kStartCal, &type);
}

void IMU::stopCalibration() {
   writeCommand(kStopCal, NULL);
}

int IMU::takeCalibrationPoint() {
   SampleCount point;
   sendCommand(kTakeUserCalSample, NULL, kUserCalSampleCount, &point);
   return point;
}

CalScore IMU::getCalibrationScore() {
   UserCalScore score;
   readCommand(kUserCalScore, &score);
   saveConfig();
   return { score.mag_cal_score, 
            score.accel_cal_score, 
            score.distribution_error, 
            score.tilt_error, 
            score.tilt_range};
}

void IMU::resetMagCalibration() {
   sendCommand(kFactoryMagCoeff, NULL, kFactoryMagCoeffDone, NULL);
   saveConfig();
}

void IMU::resetAccelCalibration() {
   sendCommand(kFactoryAccelCoeff, NULL, kFactoryAccelCoeffDone, NULL);
   saveConfig();
}

void IMU::setAHRSMode(bool mode) {
   writeCommand(kSetFunctionalMode, &mode);
   saveConfig();
}

bool IMU::getAHRSMode() {
   bool mode;
   sendCommand(kGetFunctionalMode, NULL, kGetFunctionalModeResp, &mode);
   return mode;
}

void IMU::setFIRFilters(FilterData data) {
   FIRFilter filter_id = {3,1};
   switch (data.count) {
      case F_0: {
         FIRTaps_Zero zero = {filter_id, F_0};
         sendCommand(kSetFIRFiltersZero, &zero, kSetFIRFiltersDone, NULL);
         break; }
      case F_4: {
         FIRTaps_Four four = {filter_id, F_4};
         if (!memcpy(&(four.taps),data.coeffs.get(), F_4*sizeof(double)))
            throw IMUException("Could not copy FIR coefficient array.");
         sendCommand(kSetFIRFiltersFour, &four, kSetFIRFiltersDone, NULL);
         break; }
      case F_8: {
         FIRTaps_Eight eight = {filter_id, F_8};
         if (!memcpy(&(eight.taps),data.coeffs.get(), F_8*sizeof(double)))
            throw IMUException("Could not copy FIR coefficient array.");
         sendCommand(kSetFIRFiltersEight, &eight, kSetFIRFiltersDone, NULL);
         break; }
      case F_16: {
         FIRTaps_Sixteen sixteen = {filter_id, F_16};
         if (!memcpy(&(sixteen.taps),data.coeffs.get(), F_16*sizeof(double)))
            throw IMUException("Could not copy FIR coefficient array.");
         sendCommand(kSetFIRFiltersSixteen, &sixteen, kSetFIRFiltersDone, NULL);
         break; }
      case F_32: {
         FIRTaps_ThirtyTwo thirtytwo = {filter_id, F_32};
         if (!memcpy(&(thirtytwo.taps),data.coeffs.get(), F_32*sizeof(double)))
            throw IMUException("Could not copy FIR coefficient array.");
         sendCommand(kSetFIRFiltersThirtyTwo, &thirtytwo, kSetFIRFiltersDone, NULL);
         break; }
   }
}

FilterData IMU::getFIRFilters() {
   throw IMUException("NOT IMPLEMENTED SORRY");
}

void IMU::powerDown() {
   sendCommand(kPowerDown, NULL, kPowerDownDone, NULL);
}

void IMU::wakeUp() {
   bool mode;
   writeCommand(kGetFunctionalMode, NULL);
   readCommand(kPowerUpDone, NULL);
   readCommand(kGetFunctionalMode, &mode);
}

/******************************************************************************
 * Internal Functionality
 * All of the following functions are meant for internal-use only
 ******************************************************************************/

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
 * Reads raw data from the serial port.
 * @return number of bytes not read
 */
int IMU::readRaw(void* blob, uint16_t bytes_to_read)
{  
   // Keep track of the number of bytes read, and the number of fds that are ready.
   int bytes_read = 0, current_read = 0, fds_ready = 0;
   // Sets of file descriptors for use with select(2).
   fd_set read_fds, write_fds, except_fds;
   // Timeout in the form of {sec, usec}, for use with select(2).
   struct timeval timeout = _timeout;
   // Ensure the device is avaliable and open.
   assertOpen();
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
            current_read = read(_deviceFD, (((char*)blob) + bytes_read), 
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
int IMU::writeRaw(void* blob, uint16_t bytes_to_write)
{
   // Keep track of the number of bytes written, and the number of fds that are ready.
   int bytes_written = 0, current_write = 0, fds_ready = 0;
   // Sets of file descriptors for use with select(2).
   fd_set read_fds, write_fds, except_fds;
   // Timeout in the form of {sec, usec}, for use with select(2).
   struct timeval timeout = _timeout;
   // Ensure the device is avaliable and open.
   assertOpen();
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
            current_write = write(_deviceFD, (((char*)blob) + bytes_written), 
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

#define BUF_SIZE 4096
#define ENDIAN16(A) ((A << 8) | (A >> 8))

Command IMU::readFrame(Command hint, void* target)
{
   // Temporary storage for the data read from the datagram.
   uint8_t datagram[BUF_SIZE];
   // Pointer to the bytecount
   bytecount_t *bytecount_p = reinterpret_cast<bytecount_t*>(datagram);
   // Pointer to the start of the frame
   uint8_t *frame_p = datagram + sizeof(bytecount_t);
   // Pointer to the checksum at the end of the datagram.
   checksum_t *checksum_p = NULL;
   // Some storage for data pulled out of the datagram.
   bytecount_t total_size, checksum_size, frame_size;
   // Storage for the expected checksum.
   checksum_t checksum;
   // Storage for the read command
   Command cmd;

   // Read in the header of the datagram packet: UInt16.
   if (readRaw(bytecount_p, sizeof(bytecount_t)))
      throw IMUException("Unable to read bytecount of incoming packet.");
   // Calculate the number of bytes to read from the header.
   total_size = *bytecount_p;
   // Convert from big-endian to little-endian.
   total_size = ENDIAN16(total_size);
   // Do not include the checksum in the checksum
   checksum_size = total_size - sizeof(checksum_t);
   // Do not include the bytecount in the frame.
   frame_size = checksum_size - sizeof(bytecount_t);

   // Read in the actual data frame + checksum
   if (readRaw(frame_p, frame_size + sizeof(checksum_t)))
      throw IMUException("Unable to read body of incoming packet");
   // Find the position of the checksum
   checksum_p = reinterpret_cast<checksum_t*>(frame_p + frame_size);
   // Pull out the sent checksum and convert from big-endian to little-endian.
   checksum = ENDIAN16(*checksum_p);

   // Validate the existing checksum
   if (crc16(datagram, checksum_size) != checksum)
      throw IMUException("Incoming packet checksum invalid.");

   // Copy the data into the given buffer.
   memcpy(target, frame_p + sizeof(frameid_t), cmd.payload_size);

   if (*frame_p == hint.id) {
      cmd = hint;
   } else {
      switch(*frame_p) {
         case kGetModInfoResp.id:               
            cmd = kGetModInfoResp; break;
         case kGetDataResp.id:                  
            cmd = kGetDataResp; break;
         case kGetConfigRespBoolean.id:         
            switch (frame_size) {
               case kGetConfigRespFloat32.payload_size:
                  cmd = kGetConfigRespFloat32; break;
               case kGetConfigRespUInt8.payload_size:
                  cmd = kGetConfigRespUInt8; break;
               default:
                  throw IMUException("Unknown command response packet length");
            } break;
         case kGetFIRFiltersRespZero.id:        
            switch (frame_size) {
               case kGetFIRFiltersRespZero.payload_size:
                  cmd = kGetFIRFiltersRespZero; break;
               case kGetFIRFiltersRespFour.payload_size:
                  cmd = kGetFIRFiltersRespFour; break;
               case kGetFIRFiltersRespEight.payload_size:
                  cmd = kGetFIRFiltersRespEight; break;
               case kGetFIRFiltersRespSixteen.payload_size:
                  cmd = kGetFIRFiltersRespSixteen; break;
               case kGetFIRFiltersRespThirtyTwo.payload_size:
                  cmd = kGetFIRFiltersRespThirtyTwo; break;
               default:
                  throw IMUException("Unknown filter response packet length");
            } break;
         case kSaveDone.id:                     
            cmd = kSaveDone; break;
         case kUserCalSampleCount.id:           
            cmd = kUserCalSampleCount; break;
         case kUserCalScore.id:                 
            cmd = kUserCalScore; break;
         case kSetConfigDone.id:                
            cmd = kSetConfigDone; break;
         case kSetFIRFiltersDone.id:            
            cmd = kSetFIRFiltersDone; break;
         case kPowerUpDone.id:                  
            cmd = kPowerUpDone; break;
         case kSetAcqParamsDone.id:             
            cmd = kSetAcqParamsDone; break;
         case kGetAcqParamsResp.id:             
            cmd = kGetAcqParamsResp; break;
         case kPowerDownDone.id:                
            cmd = kPowerDownDone; break;
         case kFactoryMagCoeffDone.id:          
            cmd = kFactoryMagCoeffDone; break;
         case kFactoryAccelCoeffDone.id:        
            cmd = kFactoryAccelCoeffDone; break;
         case kGetFunctionalModeResp.id:        
            cmd = kGetFunctionalModeResp; break;
         case kGetMagTruthMethod.id:            
            cmd = kGetMagTruthMethod; break;
         case kGetMagTruthMethodResp.id:        
            cmd = kGetMagTruthMethodResp; break;
         default: 
            throw IMUException("Unknown frameid read from device!"); break;
      }
   }
#ifdef DEBUG
   printf("Read command %s.\n", cmd.name);
#endif
   // Tell the caller what message we read.
   return cmd;
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
   uint8_t buffer[BUF_SIZE];
   Command recv;
   do {
      recv = readFrame(cmd,buffer);
      if (recv.id == cmd.id)
         memcpy(target, buffer, cmd.payload_size);
      else {
#ifdef DEBUG
         printf("Read unexpected command: %s\n", cmd.name);
#endif
      }
   } while (recv.id != cmd.id);
}

#undef BUF_SIZE
#undef ENDIAN16

const RawDataFields IMU::dataConfig             = {
   10, kQuaternion, kGyroX, kGyroY, kGyroZ,
   kAccelX, kAccelY, kAccelZ, kMagX, kMagY, kMagZ};

