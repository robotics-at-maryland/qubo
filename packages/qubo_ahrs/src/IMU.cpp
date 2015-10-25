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

         /* Open the serial port and store into a file descriptor.
          * O_RDWR allows for bi-directional I/O
          * O_ASYNC generates signals when data is transmitted
          * allowing for I/O blocking to be resolved.
          */
         _deviceFD = open(_deviceFile, O_RDWR, O_ASYNC);
         if (fd != -1)
         {
            // Device exists, we can configure the interface.
            tcgetattr(_deviceFD, &termcfg);

            // Set the baudrate for the terminal
            cfsetospeed(&termcfg, _termBaud);
            cfsetispeed(&termcfg, _termBaud);

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
            tcsetattr(_deviceFD, TCSANOW, &termcfg);

         }
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




















