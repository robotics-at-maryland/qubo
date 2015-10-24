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
         //TODO
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
       * Loads the configuration data to the IMU hardware.
       * @return Error code
       */
      int loadConfig();

      /**
       * Reads the data from the hardware IMU
       * @return Error code
       */
      int readIMUData();
   private:
      /** 
       * Reads raw data from the serial port.
       * @return The number of bytes read in.
       */
      int readRaw(char* blob, int bytes_to_read)
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
               // Keep reading until we've filled the buffer, or ran out of data.
            } while (bytes_read < bytes_to_read && current_read > 0);
         }
         // Return the number of bytes we actually managed to read.
         return bytes_read;
      }

      /**
       * Write raw data to the serial port.
       * @return the number of bytes written.
       */
      int writeRaw(char* blob, int bytes)
      {
         return write(_deviceFD, blob, bytes);
      }

      int writeCommand(IMUFrame frame, char* payload)
      {
         
      }
      int readDatagram()
      {
      }
      int readDataResp()
      {
      }
      int readConfigResp()
      {
      }
      int readTapsResp()
      {
      }
}




















