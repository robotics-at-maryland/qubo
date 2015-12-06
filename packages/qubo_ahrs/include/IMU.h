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
// vector type
#include <vector>
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
#include "types.h"
#include "statics.h"
#include "functions.h"
   public:
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

      /** Read bytes to a blob, return the bytes not read. */
      int readRaw(void* blob, uint16_t bytes_to_read);
      /** Write bytes from a blob, return the bytes not written. */
      int writeRaw(void* blob, uint16_t bytes_to_write);
      /** Checksum function to compute binary CRC16s. */
      checksum_t crc16(uint8_t* data, bytecount_t bytes);
      /** Checksum helper function. */
      uint16_t crc_xmodem_update (uint16_t crc, uint8_t data);

      /** Infer the command based on the frameid and bytes read. */
      Command inferCommand(Command hint, frameid_t id, bytecount_t size);
      /** Read a data packet and figure out the command. */
      Command readFrame(Command hint, void* blob);
      /** Write a command with a payload to the device. */
      void writeCommand(Command cmd, const void* payload);
      /** Read a command and its payload from the device. */
      void readCommand(Command cmd, void* target);
      /** Send a command and wait for a response. */
      void sendCommand(Command cmd, const void* payload, Command resp, void* target);
};
#endif
