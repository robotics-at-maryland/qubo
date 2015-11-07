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
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>

// Header include
#include "../include/IMU.h"

   IMU::IMU(std::string deviceFile) 
: _deviceFile(deviceFile), _deviceFD(-1), _termBaud(k38400.baud) 
{
}

IMU::~IMU()
{
   closeDevice();
}

void IMU::openDevice()
{
   struct termios termcfg;
   int modemcfg = 0, fd = -1;
   /* Open the serial port and store into a file descriptor.
    * O_RDWR allows for bi-directional I/O
    * O_ASYNC generates signals when data is transmitted
    * allowing for I/O blocking to be resolved.
    */
   fd = open(_deviceFile.c_str(), O_RDWR, O_ASYNC);
   // Check to see if the device exists.
   if (fd == -1)
      throw IMUException("Unix device '"+_deviceFile+"' not found.");
   // Read the config of the interface.
   if(tcgetattr(fd, &termcfg)) 
      throw IMUException("Unable to read terminal configuration.");

   // Set the baudrate for the terminal
   if(cfsetospeed(&termcfg, _termBaud))
      throw IMUException("Unable to set terminal output speed.");
   if(cfsetispeed(&termcfg, _termBaud))
      throw IMUException("Unable to set terminal intput speed.");

   // Configure the control modes for the terminal.
   // Replace the existing char size config to be 8 bits.
   termcfg.c_cflag = (termcfg.c_cflag & ~CSIZE) | CS8;
   // Ignore modem control lines, and enable the reciever.
   termcfg.c_cflag |= CLOCAL | CREAD;
   // Disable parity generation/checking
   termcfg.c_cflag &= ~(PARENB | PARODD);
   // Disable hardware flow control
   termcfg.c_cflag &= ~CRTSCTS;
   // Send one stop bit only.
   termcfg.c_cflag &= ~CSTOPB;

   // Configure the input modes for the terminal.
   // Ignore break condition on input. 
   termcfg.c_iflag = IGNBRK;
   // Disable X control flow, only START char restarts output.
   termcfg.c_iflag &= ~(IXON|IXOFF|IXANY);

   // Configure the local modes for the terminal.
   termcfg.c_lflag = 0;

   // Configure the output modes for the terminal.
   termcfg.c_oflag = 0;

   // Configure the read timeout (deciseconds)
   termcfg.c_cc[VTIME] = 1;
   // Configure the minimum number of chars for read.
   termcfg.c_cc[VMIN] = 60;

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
   termcfg.c_cflag &= ~CRTSCTS;
   // Push the config back to the terminal (again).
   if (tcsetattr(fd, TCSANOW, &termcfg))
      throw IMUException("Unable to re-set terminal configuration.");

   // Successful execution!
   _deviceFD = fd;
}

bool IMU::isOpen() {return _deviceFD >= 0;}

void IMU::closeDevice()
{
   // Attempt to close the device from the file descriptor.
   close(_deviceFD);
   // Clear the file descriptor 
   _deviceFD = -1;
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
   // If we havent changed anything, dont send anything.
   ConfigFloat32 float32;
   ConfigUInt32 uint32;
   ConfigUInt8 uint8;
   ConfigBoolean boolean;

   // Struct parameters can be sent directly.
   sendCommand(kSetAcqParams,                &(_stagedConfig.acqParams), 
         kSetAcqParamsDone,            NULL);
   sendCommand(kSetFIRFiltersThirtyTwo,      &(_stagedConfig.filters),
         kSetFIRFiltersDone,           NULL);
   writeCommand(kSetMagTruthMethod,          &(_stagedConfig.magTruthMethod));
   writeCommand(kSetFunctionalMode,          &(_stagedConfig.mode));
   // Primitive 
   float32.id = kDeclination,          float32.value = _stagedConfig.declination; 
   sendCommand(kSetConfigFloat32,      &float32,   kSetConfigDone,  NULL);

   uint32.id = kUserCalNumPoints,      uint32.value = _stagedConfig.userCalNumPoints;
   sendCommand(kSetConfigUInt32,       &uint32,    kSetConfigDone,   NULL);

   uint32.id = kMagCoeffSet,           uint32.value = _stagedConfig.magCoeffSet;
   sendCommand(kSetConfigUInt32,       &uint32,    kSetConfigDone,   NULL);

   uint32.id = kAccelCoeffSet,         uint32.value = _stagedConfig.accelCoeffSet;
   sendCommand(kSetConfigUInt32,       &uint32,    kSetConfigDone,   NULL);

   uint8.id = kMountingRef,            uint8.value = _stagedConfig.mountingRef;
   sendCommand(kSetConfigUInt8,        &uint8,     kSetConfigDone,    NULL);

   uint8.id = kBaudRate,               uint8.value = _stagedConfig.baudRate;
   sendCommand(kSetConfigUInt8,        &uint8,     kSetConfigDone,    NULL);

   boolean.id = kTrueNorth,            boolean.value = _stagedConfig.trueNorth;
   sendCommand(kSetConfigBoolean,      &boolean,   kSetConfigDone,  NULL);

   boolean.id = kBigEndian,            boolean.value = _stagedConfig.bigEndian;
   sendCommand(kSetConfigBoolean,      &boolean,   kSetConfigDone,  NULL);

   boolean.id = kUserCalAutoSampling,  boolean.value = _stagedConfig.userCalAutoSampling;
   sendCommand(kSetConfigBoolean,      &boolean,   kSetConfigDone,  NULL);

   boolean.id = kMilOut,               boolean.value = _stagedConfig.milOut;
   sendCommand(kSetConfigBoolean,      &boolean,   kSetConfigDone,  NULL);

   boolean.id = kHPRDuringCal,         boolean.value = _stagedConfig.hprDuringCal;
   sendCommand(kSetConfigBoolean,      &boolean,   kSetConfigDone,  NULL);
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
   sendCommand(kGetAcqParams,                NULL, 
         kGetAcqParamsResp,            &(_liveConfig.acqParams));
   sendCommand(kGetFIRFilters,               &(_liveConfig.filters), 
         kGetFIRFiltersRespThirtyTwo,  &(_liveConfig.filters));
   sendCommand(kGetMagTruthMethod,           NULL,
         kGetMagTruthMethodResp,       &(_liveConfig.magTruthMethod));
   sendCommand(kGetFunctionalMode,           NULL,
         kGetFunctionalModeResp,       &(_liveConfig.mode));
   sendCommand(kGetConfig,                   &kDeclination,
         kGetConfigRespFloat32,        &(_liveConfig.declination));
   sendCommand(kGetConfig,                   &kUserCalNumPoints,
         kGetConfigRespUInt32,         &(_liveConfig.userCalNumPoints));
   sendCommand(kGetConfig,                   &kMagCoeffSet,
         kGetConfigRespUInt32,         &(_liveConfig.magCoeffSet));
   sendCommand(kGetConfig,                   &kAccelCoeffSet,
         kGetConfigRespUInt32,         &(_liveConfig.accelCoeffSet));
   sendCommand(kGetConfig,                   &kMountingRef,
         kGetConfigRespUInt8,          &(_liveConfig.mountingRef));
   sendCommand(kGetConfig,                   &kBaudRate,
         kGetConfigRespUInt8,          &(_liveConfig.baudRate));
   sendCommand(kGetConfig,                   &kTrueNorth,
         kGetConfigRespBoolean,        &(_liveConfig.trueNorth));
   sendCommand(kGetConfig,                   &kBigEndian,
         kGetConfigRespBoolean,        &(_liveConfig.bigEndian));
   sendCommand(kGetConfig,                   &kUserCalAutoSampling,
         kGetConfigRespBoolean,        &(_liveConfig.userCalAutoSampling));
   sendCommand(kGetConfig,                   &kMilOut,
         kGetConfigRespBoolean,        &(_liveConfig.milOut));
   sendCommand(kGetConfig,                   &kHPRDuringCal,
         kGetConfigRespBoolean,        &(_liveConfig.hprDuringCal));

   // Return the configuration we just read in.
   return _liveConfig;
}

/**
 * Sends the current data format to the TRAX so we can ensure typesafety.
 * This must be done before the first pollIMUData or the data may be corrupted.
 */
void IMU::sendIMUDataFormat()
{
   writeCommand(kSetDataComponents, &dataConfig);
}

#define ENDIAN32(A) (((A>>24)&0xff) | ((A<<8)&0xff0000) | ((A>>8)&0xff00) | ((A<<24)&0xff000000))

float endianFloat32(float in)
{
   long raw = * (long *) &in;
   raw = ENDIAN32(raw);
   return * (float *) &raw;
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
   _lastReading.quaternion[0] = endianFloat32(data.quaternion[0]);
   _lastReading.quaternion[1] = endianFloat32(data.quaternion[1]);
   _lastReading.quaternion[2] = endianFloat32(data.quaternion[2]);
   _lastReading.quaternion[3] = endianFloat32(data.quaternion[3]);
   _lastReading.gyroX = endianFloat32(data.gyroX);
   _lastReading.gyroY = endianFloat32(data.gyroY);
   _lastReading.gyroZ = endianFloat32(data.gyroZ);
   _lastReading.accelX = endianFloat32(data.accelX);
   _lastReading.accelY = endianFloat32(data.accelY);
   _lastReading.accelZ = endianFloat32(data.accelZ);
   _lastReading.magX = endianFloat32(data.magX);
   _lastReading.magY = endianFloat32(data.magY);
   _lastReading.magZ = endianFloat32(data.magZ);
   return _lastReading;
}

#ifdef DEBUG
void printRaw(uint8_t* blob, uint16_t bytes)
{
   int i;
   for (i = 0; i < bytes; i++)
   {
      printf("%x ",blob[i]);
   }
   printf("\n");
}
#endif

/** 
 * Reads raw data from the serial port.
 * @return number of bytes not read
 */
int IMU::readRaw(uint8_t* blob, uint16_t bytes_to_read)
{
   // Keep track of the number of bytes read
   int bytes_read = 0, current_read = 0;
#ifdef DEBUG
   printf("Reading %d bytes\n", bytes_to_read);
#endif
   // If we need to read something, attempt to.
   while (bytes_read < bytes_to_read && current_read >= 0) {
      // Advance the pointer, and reduce the read size in each iteration.
      current_read = read(_deviceFD, 
            (blob + bytes_read), 
            (bytes_to_read - bytes_read)
            );
      bytes_read += current_read;
      // Keep reading until we've run out of data, or an error occurred.
   }
#ifdef DEBUG
   printf("Read %d bytes\n", bytes_read);
   printRaw(blob, bytes_read);
#endif
   // Return the number of bytes we actually managed to read.
   return bytes_to_read - bytes_read;
}

/**
 * Write raw data to the serial port.
 * @return number of bytes not written.
 */
int IMU::writeRaw(uint8_t* blob, uint16_t bytes_to_write)
{
   int bytes_written = 0, current_write = 0;
#ifdef DEBUG
   printf("Writing %d bytes\n", bytes_to_write);
   printRaw(blob, bytes_to_write);
#endif
   while ((bytes_written < bytes_to_write) && (current_write >= 0)){
      // Advance the pointer, and reduce the write size in each iteration.
      current_write = write(_deviceFD,
            (blob + bytes_written),
            (bytes_to_write - bytes_written)
            );
      bytes_written += current_write;
      // Keep reading until we've written everything, or an error occured.
   }
#ifdef DEBUG
   printf("Wrote %d bytes\n", bytes_to_write);
#endif
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
   if (memset(target, 0, resp.payload_size) == NULL)
      throw IMUException("Unable to clear command response target memory.");
   writeCommand(send, payload);
   readCommand(resp, target);
}

#define ENDIAN16(A) ((A << 8) | (A >> 8))

void IMU::writeCommand(Command cmd, const void* payload)
{
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

   // Copy the total datagram size to the datagram.
   *bytecount = ENDIAN16(total_size);
   // Copy the frameid from the given Command into the datagram.
   *frame = cmd.id;
   // Copy the payload to the datagram.
   memcpy(data, payload, cmd.payload_size);
   // Compute the checksum
   *checksum = crc16(datagram, checksum_size);
   *checksum = ENDIAN16(*checksum);

   // Attempt to write the datagram to the serial port.
   if (writeRaw(datagram, total_size))
      throw IMUException("Unable to write command.");
}

void IMU::readCommand(Command cmd, void* target)
{
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


















