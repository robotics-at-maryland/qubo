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

#include "impl.cpp"

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

/******************************************************************************
 * Internal Functionality
 * All of the following functions are meant for internal-use only
 ******************************************************************************/

IMU::checksum_t IMU::crc_xmodem_update (checksum_t crc, uint8_t data)
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

IMU::checksum_t IMU::crc16(checksum_t crc, uint8_t* data, bytecount_t bytes){
   for (; bytes > 0; bytes--, data++)
      crc = crc_xmodem_update(crc, *data);
   return crc;
}

int IMU::readRaw(void* blob, int bytes_to_read)
{  
#ifdef DEBUG
   printf("Reading %d bytes ", bytes_to_read);
#endif
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
#ifdef DEBUG
   int i;
   for (i = 0; i < bytes_read; i++)
      printf("|%2x",*(((unsigned char*)blob) + i));
   printf("|\n");
#endif
   // Return the number of bytes we actually managed to read.
   return bytes_to_read - bytes_read;
}

int IMU::writeRaw(void* blob, int bytes_to_write)
{
#ifdef DEBUG
   printf("Writing %d bytes ", bytes_to_write);
   int i;
   for (i = 0; i < bytes_to_write; i++)
      printf("|%2x",*(((unsigned char*)blob) + i));
   printf("|\n");
#endif
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

#define ENDIAN16(A) ((A << 8) | (A >> 8))

IMU::Message IMU::readMessage()
{
#ifdef DEBUG
   printf("Reading message\n");
#endif
   // Storage for packet bytecount.
   bytecount_t total_size;
   // Storage for calculating the checksum.
   checksum_t remote_checksum, checksum = 0x0;
   // Output structure, storage for the read memory.
   Message message;

   // Read in the header of the datagram packet: UInt16.
   if (readRaw(&total_size, sizeof(bytecount_t)))
      throw IMUException("Unable to read bytecount of incoming packet.");
   // Add the total size to the checksum.
   checksum = crc16(checksum, (uint8_t*) &total_size, sizeof(bytecount_t));
   // Convert the total size from big-endian to little-endian.
   total_size = ENDIAN16(total_size);
   // Do not include the checksum, frameid, or bytecount in the payload.
   message.payload_size = 
      total_size - sizeof(checksum_t) - sizeof(bytecount_t) - sizeof(frameid_t);

   // Create the memory in the Message struct.
   message.payload = std::make_shared<std::vector<char>>();
   message.payload->reserve(message.payload_size);

   // Read in the frameid: UInt8.
   if (readRaw(&message.id, sizeof(frameid_t)))
      throw IMUException("Unable to read frameid of incoming packet");
   // Add the message id to the checksum.
   checksum = crc16(checksum, (uint8_t*) &message.id, sizeof(frameid_t));
   // Read in the payload and spit it into the vector storage.
   if (readRaw(message.payload->data(), message.payload_size))
      throw IMUException("Unable to read payload of incoming packet");
   // Add the data read in to the checksum.
   checksum = crc16(checksum, (uint8_t*) message.payload->data(), message.payload_size);
   // Read the remote checksum that the device computed.
   if (readRaw(&remote_checksum, sizeof(checksum_t)))
      throw IMUException("Unable to read checksum of incoming packet");
   // Convert the checksum from big-endian to little-endian.
   remote_checksum = ENDIAN16(remote_checksum);

   // Validate the remote checksum
   if (checksum != remote_checksum)
      throw IMUException("Incoming packet checksum invalid.");
   
#ifdef DEBUG
   printf("Read message %d\n", message.id);
#endif

   // Everything succeeded. The packet was read in properly.
   return message;
}

void IMU::writeMessage(Message message)
{
#ifdef DEBUG
   printf("Writing message %d\n", message.id);
#endif
   // Calculate the total packet length.
   bytecount_t total_size = 
      sizeof(bytecount_t) + sizeof(frameid_t) + message.payload_size + sizeof(checksum_t);
   // Storage for the checksum.
   checksum_t checksum = 0x0;

   // Convert the packet length to big-endian for the device.
   total_size = ENDIAN16(total_size);

   // Compute the checksum from the packet data.
   checksum = crc16(checksum, (uint8_t*) &total_size, sizeof(bytecount_t));
   checksum = crc16(checksum, (uint8_t*) &message.id, sizeof(frameid_t));
   checksum = crc16(checksum, (uint8_t*) message.payload->data(), message.payload_size);
   // Convert from little-endian to big-endian
   checksum = ENDIAN16(checksum);

   // Attempt to write the datagram to the serial port.
   if (writeRaw(&total_size, sizeof(bytecount_t)))
      throw IMUException("Unable to write bytecount.");
   // Attempt to write the frameid to the serial port.
   if (writeRaw(&message.id, sizeof(frameid_t)))
      throw IMUException("Unable to write frameid.");
   // Attempt to write the payload to the serial port.
   if (writeRaw(message.payload->data(), message.payload_size))
      throw IMUException("Unable to write payload.");
   // Attempt to write the checksum to the serial port.
   if (writeRaw(&checksum, sizeof(checksum_t)))
      throw IMUException("Unable to write checksum.");
}

#undef ENDIAN16

IMU::Message IMU::createMessage(Command cmd, const void* payload)
{
#ifdef DEBUG
   printf("Creating message from %s\n", cmd.name);
#endif
   // Temporary storage to assemble the message.
   Message message;

   // Copy the details of the command to the message.
   message.id = cmd.id;
   message.payload_size = cmd.payload_size;

   // Create the memory in the Message struct.
   message.payload = std::make_shared<std::vector<char>>();
   message.payload->reserve(message.payload_size);

   // Copy the payload to the vector.
   message.payload->assign((char*) payload, ((char*)payload) + message.payload_size);

   // Return the message created.
   return message;
}

IMU::Command IMU::inferCommand(Message message)
{
#ifdef DEBUG
   printf("Inferring command from id %d\n", message.id);
#endif
   // In many cases after comparing the ID the command is unambiguous.
   // Some cases need some special attention.
   switch(message.id) {
      case kGetModInfo.id:
         return kGetModInfo;
      case kGetModInfoResp.id:
         return kGetModInfoResp;
      case kSetDataComponents.id:
         return kSetDataComponents;
      case kGetData.id:
         return kGetData;
      case kGetDataResp.id:
         return kGetDataResp;
      case kSetConfigBoolean.id:
         // Read the first byte in the payload that contains the configid.
         switch (message.payload->front()) {
            case kTrueNorth:
            case kBigEndian:
            case kUserCalAutoSampling:
            case kMilOut:
            case kHPRDuringCal:
               return kSetConfigBoolean;
            case kDeclination:
               return kSetConfigFloat32;
            case kMountingRef:
            case kBaudRate:
               return kSetConfigUInt8;
            case kUserCalNumPoints:
            case kMagCoeffSet:
            case kAccelCoeffSet:
               return kSetConfigUInt32;
            default:
               throw IMUException("Unknown configuration field id. corrupt?");
         }
      case kGetConfig.id:
         return kGetConfig;
      case kGetConfigRespBoolean.id:
         // Read the first byte in the payload that contains the configid.
         switch (message.payload->front()) {
            case kTrueNorth:
            case kBigEndian:
            case kUserCalAutoSampling:
            case kMilOut:
            case kHPRDuringCal:
               return kGetConfigRespBoolean;
            case kDeclination:
               return kGetConfigRespFloat32;
            case kMountingRef:
            case kBaudRate:
               return kGetConfigRespUInt8;
            case kUserCalNumPoints:
            case kMagCoeffSet:
            case kAccelCoeffSet:
               return kGetConfigRespUInt32;
            default:
               throw IMUException("Unknown configuration field id. corrupt?");
         }
      case kSave.id:
         return kSave;
      case kStartCal.id:
         return kStartCal;
      case kStopCal.id:
         return kStopCal;
      case kSetFIRFiltersZero.id:
         // Read the third byte in the payload that contains the number of filters.
         switch (message.payload->data()[2]) {
            case F_0:
               return kSetFIRFiltersZero;
            case F_4:
               return kSetFIRFiltersFour;
            case F_8:
               return kSetFIRFiltersEight;
            case F_16:
               return kSetFIRFiltersSixteen;
            case F_32:
               return kSetFIRFiltersThirtyTwo;
            default:
               throw IMUException("Unknown number of filters. corrupt?");
         }
      case kGetFIRFilters.id:
         return kGetFIRFilters;
      case kGetFIRFiltersRespZero.id:
         // Read the third byte in the payload that contains the number of filters.
         switch (message.payload->data()[2]) {
            case F_0:
               return kGetFIRFiltersRespZero;
            case F_4:
               return kGetFIRFiltersRespFour;
            case F_8:
               return kGetFIRFiltersRespEight;
            case F_16:
               return kGetFIRFiltersRespSixteen;
            case F_32:
               return kGetFIRFiltersRespThirtyTwo;
            default:
               throw IMUException("Unknown number of filters. corrupt?");
         }
      case kPowerDown.id:
         return kPowerDown;
      case kSaveDone.id:
         return kSaveDone;
      case kUserCalSampleCount.id:
         return kUserCalSampleCount;
      case kUserCalScore.id:
         return kUserCalScore;
      case kSetConfigDone.id:
         return kSetConfigDone;
      case kSetFIRFiltersDone.id:
         return kSetFIRFiltersDone;
      case kStartContinuousMode.id:
         return kStartContinuousMode;
      case kStopContinousMode.id:
         return kStopContinousMode;
      case kPowerUpDone.id:
         return kPowerUpDone;
      case kSetAcqParams.id:
         return kSetAcqParams;
      case kGetAcqParams.id:
         return kGetAcqParams;
      case kSetAcqParamsDone.id:
         return kSetAcqParamsDone;
      case kGetAcqParamsResp.id:
         return kGetAcqParamsResp;
      case kPowerDownDone.id:
         return kPowerDownDone;
      case kFactoryMagCoeff.id:
         return kFactoryMagCoeff;
      case kFactoryMagCoeffDone.id:
         return kFactoryMagCoeffDone;
      case kTakeUserCalSample.id:
         return kTakeUserCalSample;
      case kFactoryAccelCoeff.id:
         return kFactoryAccelCoeff;
      case kFactoryAccelCoeffDone.id:
         return kFactoryAccelCoeffDone;
      case kSetFunctionalMode.id:
         return kSetFunctionalMode;
      case kGetFunctionalMode.id:
         return kGetFunctionalMode;
      case kGetFunctionalModeResp.id:
         return kGetFunctionalModeResp;
      case kSetResetRef.id:
         return kSetResetRef;
      case kSetMagTruthMethod.id:
         return kSetMagTruthMethod;
      case kGetMagTruthMethod.id:
         return kGetMagTruthMethod;
      case kGetMagTruthMethodResp.id:
         return kGetMagTruthMethodResp;
      default: 
         throw IMUException("Unable to infer command from frameid. corrupt?"); break;
   }
}

void IMU::readCommand(Command cmd, void* target)
{
#ifdef DEBUG
   printf("Reading command %s\n", cmd.name);
#endif
   // Read until the message we receive is the one we want.
   Message message;
   do {
      message = readMessage();
   } while (cmd.name != inferCommand(message).name);

   // Copy the data to the target memory.
   if (target != NULL && !memcpy(target, message.payload->data(),message.payload_size))
      throw IMUException("Unable to copy the read command to the caller's memory.");
}

void IMU::writeCommand(Command cmd, const void* payload)
{
#ifdef DEBUG
   printf("Writing command %s\n", cmd.name);
#endif
   writeMessage(createMessage(cmd, payload));
}

void IMU::sendCommand(Command send, const void* payload, Command resp, void* target)
{
#ifdef DEBUG
   printf("Sending command %s expecting %s in response\n", send.name, resp.name);
#endif
   if ((target != NULL) && memset(target, 0, resp.payload_size) == NULL)
      throw IMUException("Unable to clear command response target memory.");
   writeCommand(send, payload);
   readCommand(resp, target);
}

constexpr IMU::Command IMU::kGetModInfo;
constexpr IMU::Command IMU::kGetModInfoResp;
constexpr IMU::Command IMU::kSetDataComponents;
constexpr IMU::Command IMU::kGetData;
constexpr IMU::Command IMU::kGetDataResp;
constexpr IMU::Command IMU::kSetConfigBoolean;
constexpr IMU::Command IMU::kSetConfigFloat32;
constexpr IMU::Command IMU::kSetConfigUInt8;
constexpr IMU::Command IMU::kSetConfigUInt32;
constexpr IMU::Command IMU::kGetConfig;
constexpr IMU::Command IMU::kGetConfigRespBoolean;
constexpr IMU::Command IMU::kGetConfigRespFloat32;
constexpr IMU::Command IMU::kGetConfigRespUInt8;
constexpr IMU::Command IMU::kGetConfigRespUInt32;
constexpr IMU::Command IMU::kSave;
constexpr IMU::Command IMU::kStartCal;
constexpr IMU::Command IMU::kStopCal;
constexpr IMU::Command IMU::kSetFIRFiltersZero;
constexpr IMU::Command IMU::kSetFIRFiltersFour;
constexpr IMU::Command IMU::kSetFIRFiltersEight;
constexpr IMU::Command IMU::kSetFIRFiltersSixteen;
constexpr IMU::Command IMU::kSetFIRFiltersThirtyTwo;
constexpr IMU::Command IMU::kGetFIRFilters;
constexpr IMU::Command IMU::kGetFIRFiltersRespZero;
constexpr IMU::Command IMU::kGetFIRFiltersRespFour;
constexpr IMU::Command IMU::kGetFIRFiltersRespEight;
constexpr IMU::Command IMU::kGetFIRFiltersRespSixteen;
constexpr IMU::Command IMU::kGetFIRFiltersRespThirtyTwo;
constexpr IMU::Command IMU::kPowerDown;
constexpr IMU::Command IMU::kSaveDone;
constexpr IMU::Command IMU::kUserCalSampleCount;
constexpr IMU::Command IMU::kUserCalScore;
constexpr IMU::Command IMU::kSetConfigDone;
constexpr IMU::Command IMU::kSetFIRFiltersDone;
constexpr IMU::Command IMU::kStartContinuousMode;
constexpr IMU::Command IMU::kStopContinousMode;
constexpr IMU::Command IMU::kPowerUpDone;
constexpr IMU::Command IMU::kSetAcqParams;
constexpr IMU::Command IMU::kGetAcqParams;
constexpr IMU::Command IMU::kSetAcqParamsDone;
constexpr IMU::Command IMU::kGetAcqParamsResp;
constexpr IMU::Command IMU::kPowerDownDone;
constexpr IMU::Command IMU::kFactoryMagCoeff;
constexpr IMU::Command IMU::kFactoryMagCoeffDone;
constexpr IMU::Command IMU::kTakeUserCalSample;
constexpr IMU::Command IMU::kFactoryAccelCoeff;
constexpr IMU::Command IMU::kFactoryAccelCoeffDone;
constexpr IMU::Command IMU::kSetFunctionalMode;
constexpr IMU::Command IMU::kGetFunctionalMode;
constexpr IMU::Command IMU::kGetFunctionalModeResp;
constexpr IMU::Command IMU::kSetResetRef;
constexpr IMU::Command IMU::kSetMagTruthMethod;
constexpr IMU::Command IMU::kGetMagTruthMethod;
constexpr IMU::Command IMU::kGetMagTruthMethodResp;

constexpr IMU::config_id_t IMU::kDeclination;
constexpr IMU::config_id_t IMU::kTrueNorth;
constexpr IMU::config_id_t IMU::kBigEndian;
constexpr IMU::config_id_t IMU::kMountingRef;
constexpr IMU::config_id_t IMU::kUserCalNumPoints;
constexpr IMU::config_id_t IMU::kUserCalAutoSampling;
constexpr IMU::config_id_t IMU::kBaudRate;
constexpr IMU::config_id_t IMU::kMilOut;
constexpr IMU::config_id_t IMU::kHPRDuringCal;
constexpr IMU::config_id_t IMU::kMagCoeffSet;
constexpr IMU::config_id_t IMU::kAccelCoeffSet;

constexpr IMU::data_id_t IMU::kPitch;
constexpr IMU::data_id_t IMU::kRoll;
constexpr IMU::data_id_t IMU::kHeadingStatus;
constexpr IMU::data_id_t IMU::kQuaternion;
constexpr IMU::data_id_t IMU::kTemperature;
constexpr IMU::data_id_t IMU::kDistortion;
constexpr IMU::data_id_t IMU::kCalStatus;
constexpr IMU::data_id_t IMU::kAccelX;
constexpr IMU::data_id_t IMU::kAccelY;
constexpr IMU::data_id_t IMU::kAccelZ;
constexpr IMU::data_id_t IMU::kMagX;
constexpr IMU::data_id_t IMU::kMagY;
constexpr IMU::data_id_t IMU::kMagZ;
constexpr IMU::data_id_t IMU::kGyroX;
constexpr IMU::data_id_t IMU::kGyroY;
constexpr IMU::data_id_t IMU::kGyroZ;

constexpr IMU::IMUSpeed IMU::k0;
constexpr IMU::IMUSpeed IMU::k2400;
constexpr IMU::IMUSpeed IMU::k4800;
constexpr IMU::IMUSpeed IMU::k9600;
constexpr IMU::IMUSpeed IMU::k19200;
constexpr IMU::IMUSpeed IMU::k38400;
constexpr IMU::IMUSpeed IMU::k57600;
constexpr IMU::IMUSpeed IMU::k115200;

constexpr IMU::RawDataFields IMU::dataConfig;

