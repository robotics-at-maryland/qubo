/*
 * IMU.cpp
 * IMU Device API implementation.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris172@gmail.com>
 * All rights reserved.
 * 
 * Adapted from earlier work of Steve Moskovchenko and Joseph Lisee (2007)
 */

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
: _deviceFile(deviceFile), _termBaud(speed.baud), _deviceFD(-1), _timeout(1), _retries(1)
{ }

IMU::~IMU() { closeDevice(); }

void IMU::openDevice()
{
   struct termios termcfg;
   int modemcfg = 0, fd = -1;
#ifdef DEBUG
   printf("IMU::openDevice()\n");
#endif
   /* Open the serial port and store into a file descriptor.
    * O_RDWR allows for bi-directional I/O
    * O_ASYNC generates signals when data is transmitted
    * allowing for I/O blocking to be resolved.
    */
   fd = open(_deviceFile.c_str(), O_RDWR, O_NONBLOCK | O_SYNC | O_DIRECT | O_NOCTTY);
   // Check to see if the device exists.
   if (fd == -1)
      throw IMUException("Unix device '"+_deviceFile+"' not found.");
   //if (fcntl(fd, F_SETFL, O_ASYNC|O_NONBLOCK))
   //   throw IMUException("Could not enable ASYNC");
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
   
   // Pull the term config (again).
   if(tcgetattr(fd, &termcfg))
      throw IMUException("Unable to re-read terminal configuration.");
   // Disable hardware flow control (again).
   //termcfg.c_cflag &= ~CRTSCTS;
   // Push the config back to the terminal (again).
   if (tcsetattr(fd, TCSANOW, &termcfg))
      throw IMUException("Unable to re-set terminal configuration.");

   // Successful execution!
   _deviceFD = fd;
}

bool IMU::isOpen() {return _deviceFD >= 0;}

void IMU::closeDevice()
{
   struct termios termcfg;
   int status = 0;
#ifdef DEBUG
   printf("IMU::closeDevice()\n");
#endif
   if (isOpen()) {
      if(tcgetattr(_deviceFD, &termcfg) || cfsetospeed(&termcfg, B0) || cfsetispeed(&termcfg, B0) || tcsetattr(_deviceFD, TCSANOW, &termcfg)) {
#ifdef DEBUG
         printf("Could not hang up terminal.\n");
#endif
      } else {
#ifdef DEBUG
         printf("Hung up successfully.\n");
#endif
      }
      // Attempt to close the device from the file descriptor.
      status = close(_deviceFD);
   }
   // Clear the file descriptor 
   _deviceFD = -1;
#ifdef DEBUG
   if (status) printf("Close error!");
#endif
}

bool IMU::changeSpeed(IMUSpeed speed)
{
   // Create the baudrate payload.
   ConfigUInt8 baudRate = {kBaudRate, speed.id};
   // Place to store the save error code
   SaveError err;
   // Check to make sure the IMU is connected first.
   if (!isOpen())
      throw IMUException("IMU must be connected to change baudrate.");
   // Tell the IMU the new baudrate to expect.
   sendCommand(kSetConfigUInt8, &baudRate, kSetConfigDone, NULL);
   // Save the configuration change.
   sendCommand(kSave, NULL, kSaveDone, &err);
   return !err;
}

std::string IMU::getInfo()
{
   // Create some temporary storage for the query.
   ModInfo info;
   // Ensure the device is open and avaliable.
   if (!isOpen()) return "";
   // Get the device name/info
   sendCommand(kGetModInfo, NULL, kGetModInfoResp, &info);
   // Build a string to return
   return std::string(((char*) &info), sizeof(ModInfo));
}

/**
 * Sends the configuration data to the IMU hardware.
 */
void IMU::sendConfig()
{
   // Read in the live configuration data
   readConfig();

   // Struct parameters can be sent directly.
   sendCommand(kSetAcqParams, &(_stagedConfig.acqParams), 
         kSetAcqParamsDone, NULL);
   sendCommand(kSetFIRFiltersThirtyTwo, &(_stagedConfig.filters), 
         kSetFIRFiltersDone, NULL);

   writeCommand(kSetMagTruthMethod, &(_stagedConfig.magTruthMethod));
   writeCommand(kSetFunctionalMode, &(_stagedConfig.mode));

   sendCommand(kSetConfigFloat32, &(_stagedConfig.declination), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigUInt32, &(_stagedConfig), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigUInt32, &(_stagedConfig), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigUInt32, &(_stagedConfig.accelCoeffSet), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigUInt8, &(_stagedConfig.mountingRef), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigUInt8, &(_stagedConfig.baudRate), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigBoolean, &(_stagedConfig.trueNorth), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigBoolean, &(_stagedConfig.bigEndian), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigBoolean, &(_stagedConfig.userCalAutoSampling), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigBoolean, &(_stagedConfig.milOut), 
         kSetConfigDone, NULL);
   sendCommand(kSetConfigBoolean, &(_stagedConfig.hprDuringCal), 
         kSetConfigDone, NULL);
}

/**
 * Reads the configuration data from the IMU hardware into the live config.
 * @return the live configuration.
 */
IMUConfig IMU::readConfig()
{
   // Ensure the device is open and avaliable.
   if (!isOpen())
      throw IMUException("Device is closed");

   // Read in all config data one-by-one
   sendCommand(kGetAcqParams, NULL, 
         kGetAcqParamsResp, &(_liveConfig.acqParams));
   sendCommand(kGetFIRFilters, &kFilterID, 
         kGetFIRFiltersRespThirtyTwo, &(_liveConfig.filters));
   sendCommand(kGetMagTruthMethod, NULL, 
         kGetMagTruthMethodResp, &(_liveConfig.magTruthMethod));
   sendCommand(kGetFunctionalMode, NULL, 
         kGetFunctionalModeResp, &(_liveConfig.mode));
   sendCommand(kGetConfig, &kDeclination, 
         kGetConfigRespFloat32, &(_liveConfig.declination));
   sendCommand(kGetConfig, &kUserCalNumPoints, 
         kGetConfigRespUInt32, &(_liveConfig.userCalNumPoints));
   sendCommand(kGetConfig, &kMagCoeffSet, 
         kGetConfigRespUInt32, &(_liveConfig.magCoeffSet));
   sendCommand(kGetConfig,  &kAccelCoeffSet, 
         kGetConfigRespUInt32, &(_liveConfig.accelCoeffSet));
   sendCommand(kGetConfig, &kMountingRef, 
         kGetConfigRespUInt8, &(_liveConfig.mountingRef));
   sendCommand(kGetConfig, &kBaudRate, 
         kGetConfigRespUInt8, &(_liveConfig.baudRate));
   sendCommand(kGetConfig, &kTrueNorth, 
         kGetConfigRespBoolean, &(_liveConfig.trueNorth));
   sendCommand(kGetConfig, &kBigEndian, 
         kGetConfigRespBoolean, &(_liveConfig.bigEndian));
   sendCommand(kGetConfig, &kUserCalAutoSampling, 
         kGetConfigRespBoolean, &(_liveConfig.userCalAutoSampling));
   sendCommand(kGetConfig, &kMilOut, 
         kGetConfigRespBoolean, &(_liveConfig.milOut));
   sendCommand(kGetConfig, &kHPRDuringCal, 
         kGetConfigRespBoolean, &(_liveConfig.hprDuringCal));

   // Return the configuration we just read in.
   return _liveConfig;
}

/**
 * Sends the current data format to the TRAX so we can ensure typesafety.
 * This must be done before the first pollIMUData or the data may be corrupted.
 */
void IMU::sendIMUDataFormat()
{
   // Tell the IMU that we want little-endian data out so we dont have to convert.
   ConfigBoolean boolean = {kBigEndian, 0};
   // Send the command and wait for a response.
   sendCommand(kSetConfigBoolean,      &boolean,   kSetConfigDone,  NULL);
   // Now upload the output data configuration, without any success confirmation.
   writeCommand(kSetDataComponents, &dataConfig);
}

/**
 * Reads the data from the hardware IMU
 * @return Error code
 */

IMUData IMU::pollIMUData()
{
   // Allocate temporary storage to read in the raw data.
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

/** 
 * Reads raw data from the serial port.
 * @return number of bytes not read
 */
int IMU::readRaw(uint8_t* blob, uint16_t bytes_to_read)
{  
   // Keep track of the number of bytes read, and the select status.
   int bytes_read = 0, status = 1;
   // Sets of file descriptors for use with select(2).
   fd_set read_fds, write_fds, except_fds;
   // Timeout info struct for use with select(2).
   struct timeval timeout;
   timeout.tv_sec = _timeout;
   timeout.tv_usec = 0; 
   // If we need to read something, attempt to.
   // Keep reading until we've run out of data, or an error occurred.
   while (bytes_read < bytes_to_read && status > 0) {
      FD_ZERO(&read_fds);
      FD_ZERO(&write_fds);
      FD_ZERO(&except_fds);
      FD_SET(_deviceFD, &read_fds);
      status = select(_deviceFD+1, &read_fds, &write_fds, &except_fds, &timeout);
      if (status == 1) {
         // The filedescriptor is ready to read.
         status = read(_deviceFD, 
               (blob + bytes_read), 
               (bytes_to_read - bytes_read)
               );
         if (status > 0) {
            bytes_read += status;
         }
      } else {
         // Read timed out and we couldnt resolve the block.
#ifdef DEBUG
         if (status)
            printf("Read Error after %d bytes.\n", bytes_read);
         else
            printf("Read Timeout after %d bytes.\n", bytes_read);
#endif
      }
   }
   if (bytes_read) {
#ifdef DEBUG
      printf("Read %d bytes\n", bytes_read);
#endif
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
   // Keep track of the number of bytes read, and the select status.
   int bytes_written = 0, status = 1;
   // Sets of file descriptors for use with select(2).
   fd_set read_fds, write_fds, except_fds;
   // Timeout info struct for use with select(2).
   struct timeval timeout;
   timeout.tv_sec = _timeout;
   timeout.tv_usec = 0; 
#ifdef DEBUG
   printf("Writing %d bytes\n", bytes_to_write);
#endif
   // If we need to read something, attempt to.
   // Keep reading until we've run out of data, or an error occurred.
   while (bytes_written < bytes_to_write && status > 0) {
      FD_ZERO(&read_fds);
      FD_ZERO(&write_fds);
      FD_ZERO(&except_fds);
      FD_SET(_deviceFD, &write_fds);
      status = select(_deviceFD+1, &read_fds, &write_fds, &except_fds, &timeout);
      if (status == 1) {
         // The filedescriptor is ready to read.
         status = write(_deviceFD, 
               (blob + bytes_written), 
               (bytes_to_write - bytes_written)
               );
         if (status > 0) {
            bytes_written += status;
         }/*
         if (status != -1) {
#ifdef DEBUG
            printf("Draining the write buffer.\n");
#endif
            status = tcdrain(_deviceFD);
#ifdef DEBUG
            if (status) {
               printf("Stdrain error.\n");
            } else {
               printf("Drained properly.\n");
            }
#endif
         }*/
      } else {
         // Read timed out and we couldnt resolve the block.
#ifdef DEBUG
         if (status)
            printf("Write Error after %d bytes.\n", bytes_written);
         else
            printf("Write Timeout after %d bytes.\n", bytes_written);
#endif
      }
   }
   //Return the number of bytes not written.
   return bytes_to_write - bytes_written;
}

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
   int fail = 1;
   if ((target != NULL) && memset(target, 0, resp.payload_size) == NULL)
      throw IMUException("Unable to clear command response target memory.");
   // Send the command, if this fails then just exit.
   writeCommand(send, payload);
   while (fail) {
      try {
         readCommand(resp, target);
         fail = 0;
      } catch (IMUException& e) {
#ifdef DEBUG
         printf("Failed to read/write (#%d) %s\n", fail, e.what());
#endif
         fail++; 
      }
      if (fail > _retries)
         throw IMUException("Failed to send command too many times.");
   }
}

#define ENDIAN16(A) ((A << 8) | (A >> 8))

void IMU::writeCommand(Command cmd, const void* payload)
{
#ifdef DEBUG
   printf("Writing command ");
   printCommand(cmd);
   printf("\n");
#endif
   // Temporary storage to assemble the data packet.
   uint8_t datagram[4096];

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
   printf("Reading command ");
   printCommand(cmd);
   printf("\n");
#endif
   // Temporary storage for the data read from the datagram.
   uint8_t datagram[4096];
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

#undef ENDIAN16

void IMU::printCommand(Command cmd)
{
   switch(cmd.id) {
      case 0x01:
         printf("kGetModInfo"); break;
      case 0x02:
         printf("kGetModInfoResp"); break;
      case 0x03:
         printf("kSetDataComponents"); break;
      case 0x04:
         printf("kGetData"); break;
      case 0x05:
         printf("kGetDataResp"); break;
      case 0x06:
         printf("kSetConfig"); break;
      case 0x07:
         printf("kGetConfig"); break;
      case 0x08:
         printf("kGetConfigResp"); break;
      case 0x09:
         printf("kSave"); break;
      case 0x0a:
         printf("kStartCal"); break;
      case 0x0b:
         printf("kStopCal"); break;
      case 0x0c:
         printf("kSetFIRFilters"); break;
      case 0x0d:
         printf("kGetFIRFilters"); break;
      case 0x0e:
         printf("kGetFIRFiltersResp"); break;
      case 0x0f:
         printf("kPowerDown"); break;
      case 0x10:
         printf("kSaveDone"); break;
      case 0x11:
         printf("kUserCalSampleCount"); break;
      case 0x12:
         printf("kUserCalScore"); break;
      case 0x13:
         printf("kSetConfigDone"); break;
      case 0x14:
         printf("kSetFIRFiltersDone"); break;
      case 0x15:
         printf("kStartContinuousMode"); break;
      case 0x16:
         printf("kStopContinousMode"); break;
      case 0x17:
         printf("kPowerUpDone"); break;
      case 0x18:
         printf("kSetAcqParams"); break;
      case 0x19:
         printf("kGetAcqParams"); break;
      case 0x1a:
         printf("kSetAcqParamsDone"); break;
      case 0x1b:
         printf("kGetAcqParamsResp"); break;
      case 0x1c:
         printf("kPowerDownDone"); break;
      case 0x1d:
         printf("kFactoryMagCoeff"); break;
      case 0x1e:
         printf("kFactoryMagCoeffDone"); break;
      case 0x1f:
         printf("kTakeUserCalSample"); break;
      case 0x24:
         printf("kFactoryAccelCoeff"); break;
      case 0x25:
         printf("kFactoryAccelCoeffDone"); break;
      case 0x4f:
         printf("kSetFunctionalMode"); break;
      case 0x50:
         printf("kGetFunctionalMode"); break;
      case 0x51:
         printf("kGetFunctionalModeResp"); break;
      case 0x6e:
         printf("kSetResetRef"); break;
      case 0x77:
         printf("kSetMagTruthMethod"); break;
      case 0x78:
         printf("kGetMagTruthMethod"); break;
      case 0x79:
         printf("kGetMagTruthMethodResp"); break;
      default:
         printf("kINVALID");
   }
}

/******************************************************************************
 * Below is all hardcoded data from the protocol spec'd for the TRAX.
 *            | Command Name               | ID   | Payload Size
 ******************************************************************************/
const Command IMU::kGetModInfo                  = {0x01, EMPTY                      };
const Command IMU::kGetModInfoResp              = {0x02, sizeof(ModInfo)            };
const Command IMU::kSetDataComponents           = {0x03, sizeof(RawDataFields)      };
const Command IMU::kGetData                     = {0x04, EMPTY                      };
const Command IMU::kGetDataResp                 = {0x05, sizeof(RawData)            };
const Command IMU::kSetConfigBoolean            = {0x06, sizeof(ConfigBoolean)      };
const Command IMU::kSetConfigFloat32            = {0x06, sizeof(ConfigFloat32)      };
const Command IMU::kSetConfigUInt8              = {0x06, sizeof(ConfigUInt8)        };
const Command IMU::kSetConfigUInt32             = {0x06, sizeof(ConfigUInt32)       };
const Command IMU::kGetConfig                   = {0x07, sizeof(config_id_t)        };
const Command IMU::kGetConfigRespBoolean        = {0x08, sizeof(ConfigBoolean)      };
const Command IMU::kGetConfigRespFloat32        = {0x08, sizeof(ConfigFloat32)      };
const Command IMU::kGetConfigRespUInt8          = {0x08, sizeof(ConfigUInt8)        };
const Command IMU::kGetConfigRespUInt32         = {0x08, sizeof(ConfigUInt32)       };
const Command IMU::kSave                        = {0x09, EMPTY                      };
const Command IMU::kStartCal                    = {0x0a, sizeof(CalOption)          };
const Command IMU::kStopCal                     = {0x0b, EMPTY                      };
const Command IMU::kSetFIRFiltersZero           = {0x0c, sizeof(FIRTaps_Zero)       };
const Command IMU::kSetFIRFiltersFour           = {0x0c, sizeof(FIRTaps_Four)       };
const Command IMU::kSetFIRFiltersEight          = {0x0c, sizeof(FIRTaps_Eight)      };
const Command IMU::kSetFIRFiltersSixteen        = {0x0c, sizeof(FIRTaps_Sixteen)    };
const Command IMU::kSetFIRFiltersThirtyTwo      = {0x0c, sizeof(FIRTaps_ThirtyTwo)  };
const Command IMU::kGetFIRFilters               = {0x0d, sizeof(FIRFilter)          };
const Command IMU::kGetFIRFiltersRespZero       = {0x0e, sizeof(FIRTaps_Zero)       };
const Command IMU::kGetFIRFiltersRespFour       = {0x0e, sizeof(FIRTaps_Four)       };
const Command IMU::kGetFIRFiltersRespEight      = {0x0e, sizeof(FIRTaps_Eight)      };
const Command IMU::kGetFIRFiltersRespSixteen    = {0x0e, sizeof(FIRTaps_Sixteen)    };
const Command IMU::kGetFIRFiltersRespThirtyTwo  = {0x0e, sizeof(FIRTaps_ThirtyTwo)  };
const Command IMU::kPowerDown                   = {0x0f, EMPTY                      };
const Command IMU::kSaveDone                    = {0x10, sizeof(SaveError)          };
const Command IMU::kUserCalSampleCount          = {0x11, sizeof(SampleCount)        };
const Command IMU::kUserCalScore                = {0x12, sizeof(UserCalScore)       };
const Command IMU::kSetConfigDone               = {0x13, EMPTY                      };
const Command IMU::kSetFIRFiltersDone           = {0x14, EMPTY                      };
const Command IMU::kStartContinuousMode         = {0x15, EMPTY                      };
const Command IMU::kStopContinousMode           = {0x16, EMPTY                      };
const Command IMU::kPowerUpDone                 = {0x17, EMPTY                      };
const Command IMU::kSetAcqParams                = {0x18, sizeof(AcqParams)          };
const Command IMU::kGetAcqParams                = {0x19, EMPTY                      };
const Command IMU::kSetAcqParamsDone            = {0x1a, EMPTY                      };
const Command IMU::kGetAcqParamsResp            = {0x1b, sizeof(AcqParams)          };
const Command IMU::kPowerDownDone               = {0x1c, EMPTY                      };
const Command IMU::kFactoryMagCoeff             = {0x1d, EMPTY                      };
const Command IMU::kFactoryMagCoeffDone         = {0x1e, EMPTY                      };
const Command IMU::kTakeUserCalSample           = {0x1f, EMPTY                      };
const Command IMU::kFactoryAccelCoeff           = {0x24, EMPTY                      };
const Command IMU::kFactoryAccelCoeffDone       = {0x25, EMPTY                      };
const Command IMU::kSetFunctionalMode           = {0x4f, sizeof(FunctionalMode)     };
const Command IMU::kGetFunctionalMode           = {0x50, EMPTY                      };
const Command IMU::kGetFunctionalModeResp       = {0x51, sizeof(FunctionalMode)     };
const Command IMU::kSetResetRef                 = {0x6e, EMPTY                      };
const Command IMU::kSetMagTruthMethod           = {0x77, sizeof(MagTruthMethod)     };
const Command IMU::kGetMagTruthMethod           = {0x78, EMPTY                      };
const Command IMU::kGetMagTruthMethodResp       = {0x79, sizeof(MagTruthMethod)     };

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

const IMUSpeed IMU::k0                          = {0,    B0};
const IMUSpeed IMU::k2400                       = {4,    B2400};
/*const IMUSpeed IMU::k3600                       = {5,    B3600};*/
const IMUSpeed IMU::k4800                       = {6,    B4800};
/*const IMUSpeed IMU::k7200                       = {7,    B7200};*/
const IMUSpeed IMU::k9600                       = {8,    B9600};
/*const IMUSpeed IMU::k14400                      = {9,    B14400};*/
const IMUSpeed IMU::k19200                      = {10,   B19200};
/*const IMUSpeed IMU::k28800                      = {11,   B28800};*/
const IMUSpeed IMU::k38400                      = {12,   B38400};
const IMUSpeed IMU::k57600                      = {13,   B57600};
const IMUSpeed IMU::k115200                     = {14,   B115200};

const FIRFilter IMU::kFilterID                  = {3,1};
const RawDataFields IMU::dataConfig             = {
   10, kQuaternion, kGyroX, kGyroY, kGyroZ,
   kAccelX, kAccelY, kAccelZ, kMagX, kMagY, kMagZ};


















