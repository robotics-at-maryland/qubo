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

// Error handling
#include <errno.h>

// Header include
#include "IMU.h"

class IMU
{
   public:
      /**
       * Constructs a new IMU object.
       * @param deviceFile The unix serial port of the IMU hardware.
       */
      IMU(std::string deviceFile)
         : _deviceFile(deviceFile)
           _deviceFD(-1)
      {
      }
      
      /**
       * Deconstructs the IMU, cleans up.
       */
      ~IMU()
      {
         closeDevice();
      };
      
      /**
       * Opens the serial port to communicate with the IMU
       * @return Error code
       */
      int openDevice()
      {
         struct termios termcfg;
         int modemcfg = 0;
         int if(0;

         /* Open the serial port and store into a file descriptor.
          * O_RDWR allows for bi-directional I/O
          * O_ASYNC generates signals when data is transmitted
          * allowing for I/O blocking to be resolved.
          */
         _deviceFD = open(_deviceFile, O_RDWR, O_ASYNC);
         if (fd != -1)
         {
            // Device exists, we can configure the interface.
            if(tcgetattr(_deviceFD, &termcfg)) return 1;
   
            // Set the baudrate for the terminal
            if(cfsetospeed(&termcfg, _termBaud)) return 2;
            if(cfsetispeed(&termcfg, _termBaud)) return 3;

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
            if(tcsetattr(_deviceFD, TCSANOW, &termcfg)) return 4;

            // Pull in the modem configuration
            if(ioctl(_deviceFD, TIOCMGET, &modemcfg)) return 5;
            // Enable Request to Send
            modemcfg |= TIOCM_RTS;
            // Push the modem config back to the modem.
            if(ioctl(_deviceFD, TIOCMSET, &modemcfg)) return 6;

            // Pull the term config (again).
            if(tcgetattr(_deviceFD, &termcfg)) return 7;
            // Disable hardware flow control (again).
            termcfg.c_cflag &= ~CRTSCTS;
            // Push the config back to the terminal (again).
            if (tcsetattr(_deviceFD, &termcfg)) return 8;

            // Successful execution!
            return 0;
         }

         // Couldnt open the device
         return -1;
      }

      /**
       * Checks if the device is currently open.
       * @return true if the device is active.
       */
      bool isOpen() {_deviceFD >= 0};

      /**
       * Closes the serial port if it is currently connected.
       */
      void closeDevice()
      {
         // Attempt to close the device from the file descriptor.
         if (!close(_deviceFD)) {
            _deviceFD = -1;
         }
      }
      /**
       * Sends the configuration data to the IMU hardware.
       * @return Error code
       */
      int sendConfig()
      {
         // Read in the live configuration data
         readConfig();
         // If we havent changed anything, dont send anything.
         if (_stagedConfig != _liveConfig){
            ConfigFloat32 float32;
            ConfigUInt32 uint32;
            ConfigUInt8 uint8;
            ConfigBoolean boolean;
            
            // Struct parameters can be sent directly.
            sendCommand(kSetAcqParams,                &(_stagedConfig.acqParams), 
                        kSetAcqParamsDone,            NULL);
            sendCommand(kSetFIRFiltersThirtyTwo,      &(_stagedConfig.filters),
                        kSetFIRFiltersDone,           NULL);
            sendCommand(kSetMagTruthMethod,           &(_stagedConfig.magTruthMethod);
                        kSetMagTruthMethodDone,       NULL);
            sendCommand(kSetFunctionalMode,           &(_stagedConfig.mode),
                        kSetFunctionalModeDone,       NULL);
            // Primitive 
            float32.id = kDeclination,       float32.value = _stagedConfig.declination; 
            sendCommand(kSetConfigFloat32,   &float32,   kSetConfigRespFloat32,  NULL);

            uint32.id = kUserCalNumPoints,   uint32.value = _stagedConfig.userCalNumPoints;
            sendCommand(kSetConfigUInt32,    &uint32,    kSetConfigRespUInt32,   NULL);

            uint32.id = kMagCoeffSet,        uint32.value = _stagedConfig.magCoeffSet
            sendCommand(kSetConfigUInt32,    &uint32,    kSetConfigRespUInt32,   NULL);

            uint32.id = kAccelCoeffSet,      uint32.value = _stagedConfig.accelCoeffSet;
            sendCommand(kSetConfigUInt32,    &uint32,    kSetConfigRespUInt32,   NULL);

            uint8.id = kMountingRef,         uint8.value = _stagedConfig.mountingRef;
            sendCommand(kSetConfigUInt8,     &uint8,     kSetConfigRespUInt8,    NULL);

            uint8.id = kBaudRate,            uint8.value = _stagedConfig.baudRate;
            sendCommand(kSetConfigUInt8,     &uint8,     kSetConfigRespUInt8,    NULL);

            boolean.id = kTrueNorth,         boolean.value = _stagedConfig.trueNorth;
            sendCommand(kSetConfigBoolean,   &boolean,   kSetConfigRespBoolean,  NULL);

            boolean.id = kBigEndian,         boolean.value = _stagedConfig.bigEndian;
            sendCommand(kSetConfigBoolean,   &boolean,   kSetConfigRespBoolean,  NULL);

            boolean.id = kUserCalAutoSample, boolean.value = _stagedConfig.userCalAutoSampling;
            sendCommand(kSetConfigBoolean,   &boolean,   kSetConfigRespBoolean,  NULL);

            boolean.id = kMilOut,            boolean.value = _stagedConfig.milOut;
            sendCommand(kSetConfigBoolean,   &boolean,   kSetConfigRespBoolean,  NULL);

            boolean.id = kHPRDuringCal,      boolean.value = _stagedConfig.hprDuringCal;
            sendCommand(kSetConfigBoolean,   &boolean,   kSetConfigRespBoolean,  NULL);
         }
         // If the new config matches the old one, success!
         return (stagedConfig != readConfig());
      }

      /**
       * Reads the configuration data from the IMU hardware into the live config.
       * @return the live configuration.
       */
      IMUConfig readConfig()
      {
         sendCommand(kGetAcqParams,                NULL, 
               kGetAcqParamsResp,            &(_liveConfig.acqParams));
         sendCommand(kGetFIRFilters,               &firFilters, 
               kGetFIRFiltersRespThirtyTwo,  &(_liveConfig.filters));
         sendCommand(kGetMagTruthMethod,           NULL,
               kGetMagTruthMethodResp,       &(_liveConfig.magTruthMethod));
         sendCommand(kGetFunctionalMode,           NULL,
               kGetFunctionalModeResp,       &(_liveConfig.mode));
         sendCommand(kGetConfigFloat32,            &kDeclination,
               kGetConfigRespFloat32,        &(_liveConfig.declination));
         sendCommand(kGetConfigUInt32,             &kUserCalNumPoints,
               kGetConfigRespUInt32,         &(_liveConfig.userCalNumPoints));
         sendCommand(kGetConfigUInt32,             &kMagCoeffSet,
               kGetConfigRespUInt32,         &(_liveConfig.magCoeffSet));
         sendCommand(kGetConfigUInt32,             &kAccelCoeffSet,
               kGetConfigRespUInt32,         &(_liveConfig.accelCoeffSet));
         sendCommand(kGetConfigUInt8,              &kMountingRef,
               kGetConfigRespUInt8,          &(_liveConfig.mountingRef));
         sendCommand(kGetConfigUInt8,              &kBaudRate,
               kGetConfigRespUInt8,          &(_liveConfig.baudRate));
         sendCommand(kGetConfigBoolean,            &kTrueNorth,
               kGetConfigRespBoolean,        &(_liveConfig.trueNorth));
         sendCommand(kGetConfigBoolean,            &kBigEndian,
               kGetConfigRespBoolean,        &(_liveConfig.bigEndian));
         sendCommand(kGetConfigBoolean,            &kUserCalAutoSampling,
               kGetConfigRespBoolean,        &(_liveConfig.userCalAutoSampling));
         sendCommand(kGetConfigBoolean,            &kMilOut,
               kGetConfigRespBoolean,        &(_liveConfig.milOut));
         sendCommand(kGetConfigBoolean,            &kHPRDuringCal,
               kGetConfigRespBoolean,        &(_liveConfig.hprDuringCal));
      }

      /**
       * Sends the current data format to the TRAX so we can ensure typesafety.
       * This must be done before the first pollIMUData or the data may be corrupted.
       */
      int sendIMUDataFormat()
      {
         writeCommand(kSetDataComponents, &dataConfig);
      }

      /**
       * Reads the data from the hardware IMU
       * @return Error code
       */

      IMUData pollIMUData()
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
   private:
      /** 
       * Reads raw data from the serial port.
       * @return The number of bytes read in.
       */
      int readRaw(uint8_t* blob, uint16_t bytes_to_read)
      {
         // Keep track of the number of bytes read
         int bytes_read = 0, current_read = 0;
         // If we need to read something, attempt to.
         if (bytes_to_read > 0) {
            do {
               // Advance the pointer, and reduce the read size in each iteration.
               current_read = read(_deviceFD, 
                     (blob + bytes_read), 
                     (bytes_to_read - bytes_read)
                     );
               // Keep reading until we've run out of data, or an error occurred.
            } while (bytes_read < bytes_to_read && current_read >= 0);
         }
         // Return the number of bytes we actually managed to read.
         return bytes_read;
      }

      /**
       * Write raw data to the serial port.
       * @return the number of bytes written.
       */
      int writeRaw(uint8_t* blob, uint16_t bytes_to_write)
      {
         uint16_t bytes_written, current_write = 0;

         if (bytes > 0) {
            do {
               // Advance the pointer, and reduce the write size in each iteration.
               current_write = write(_deviceFD,
                     (blob + bytes_written),
                     (bytes_to_write - bytes_written)
                     );
               // Keep reading until we've written everything, or an error occured.
            } while (bytes_written < bytes_to_write & current_write >= 0);
         }
         //Return the number of bytes ultimately written.
         return bytes_written;
      }

      int writeCommand(Command cmd, void* payload)
      {
         // Temporary storage to assemble the data packet.
         uint8_t datagram[4096];
         // Pointers to specific parts of the datagram.
         bytecount_t* bytecount = (*bytecount_t) datagram;
         uint8_t* frame = datagram + sizeof(bytecount_t);
         uint8_t* data = frame + sizeof(frameid_t);
         checksum_t* checksum = payload + cmd.payload_size;
         // Various sizes of the parts of the datagram.
         bytecount_t data_offset = sizeof(bytecount_t) + sizeof(frameid_t);
         bytecount_t checksum_size = payload_offset + cmd.payload_size;
         bytecount_t total_size = checksum_size + sizeof(checksum_t);

         // Copy the total datagram size to the datagram.
         *bytecount = total_size;
         // Copy the frameid from the given Command into the datagram.
         *frame = cmd.id;
         // Copy the payload to the datagram.
         memcpy(data, payload, cmd.payload_size);
         // Compute the checksum
         *checksum = crc16(datagram, checksum_size);

         // Attempt to write the datagram to the serial port.
         if (writeRaw(datagram, byte_count) != byte_count) return 1;

         // Writing the command went well, return success.
         return 0;
      }

      /**
       * Sends a command to the device and waits for the response.
       */
      int sendCommand(Command send, void* payload, Command resp, void* target)
      {
         if (memset(target, 0, resp.payload_size) == NULL) return 1;
         if (writeCommand(send, payload)) return 2;
         if (readCommand(resp, target)) return 3;
         return 0;
      }

      int readCommand(Command cmd, void* target)
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
         if (readRaw(datagram, sizeof(bytecount_t))) return 1;
         // Calculate the number of bytes to read from the header.
         total_size = *((*bytecount_t) datagram);
         // Do not include the checksum in the checksum
         checksum_size = total+size - sizeof(checksum_t);
         // Do not include the bytecount in the frame.
         frame_size = checksum_size - sizeof(bytecount_t);

         // Read in the actual data frame + checksum
         if (readRaw(frame, frame_size + sizeof(checksum_t))) return 2;
         // Pull out the sent checksum
         checksum = *((*checksum_t)(frame + frame_size));

         // Validate the existing checksum
         if (crc16(datagram, checksum_size) == checksum) return 3;
         //Identify that the message recieved matches what was expected.
         if (*frame != cmd.id) return 4;

         // Copy the data into the given buffer.
         memcpy(target, frame, cmd.payload_size);
         // Successful datagram read.
         return 0;
      }

      checksum_t crc16(uint8_t* data, bytecount_t bytes){
         uint16_t crc;
         for (crc = 0x0; bytes > 0; bytes--, data++){
            crc = _crc_xmodem_update(crc, *data);
         }
         return (checksum_t) crc;
      }

}




















