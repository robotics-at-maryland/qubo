#ifndef AHRS_H
#define AHRS_H

/*
 * AHRS.h
 * Header file for AHRS API.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris1727@gmail.com>
 * All rights reserved.
 * 
 * Adapted from earlier work of Steve Moskovchenko and Joseph Lisee (2007)
 */

// memcpy method
#include <string>
// std::string type
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
// shared_ptr type
#include <memory>

/**
 * Exception class for handling IO/Data integrity errors.
 */
class AHRSException : public std::runtime_error
{
   public:
      AHRSException(std::string message)
         : runtime_error(message) {}
};

/**
 * AHRS API class
 * Contains all the low-level I/O abstraction, and allows the user to communicate
 * with the AHRS hardware at a high level. Connects to a unix serial/usb terminal
 * and speaks the PNI Binary protocol to send and recieve packets of data.
 */
class AHRS
{
#include "types.h"
#include "statics.h"
#include "functions.h"
   public:
      /**
       * Constructor for a new AHRS interface.
       * @param (std::string) unix device name
       * @param (AHRSSpeed) Baudrate to use for connection.
       */
      AHRS(std::string deviceFile, AHRSSpeed speed);
      /** Destructor that cleans up and closes the device. */
      ~AHRS();
      /** 
       * Opens the device and configures the I/O terminal. 
       * Requires dialout permissions to the device in _deviceFile.
       */
      void openDevice();
      /** 
       * Checks if the AHRS is currently open and avaiable 
       * @return (bool) whether the AHRS is avaliable for other API operations.
       */
      bool isOpen();
      /** Ensures that the AHRS is open, throws an exception otherwise. */
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
      /** Storage for readings from the AHRS for caching purposes. */
      AHRSData _lastReading;

      /** Checksum helper function. */
      checksum_t crc_xmodem_update (checksum_t crc, uint8_t data);
      /** Checksum function to compute binary CRC16s. */
      checksum_t crc16(checksum_t crc, uint8_t* data, bytecount_t bytes);
      /** Read bytes to a blob, return the bytes not read. */
      int readRaw(void* blob, int bytes_to_read);
      /** Write bytes from a blob, return the bytes not written. */
      int writeRaw(void* blob, int bytes_to_write);

      /** Read an incoming frame and format it for interpreting. */
      Message readMessage();
      /** Write a message to the device, adding bytecount and checksum. */
      void writeMessage(Message message);
      /** Create a message from a command and a payload */
      Message createMessage(Command cmd, const void* payload);
      /** Infer the command that the message refers to. */
      Command inferCommand(Message message);

      /** Read a command and its payload from the device. */
      void readCommand(Command cmd, void* target);
      /** Write a command with a payload to the device. */
      void writeCommand(Command cmd, const void* payload);
      /** Send a command and wait for a response. */
      void sendCommand(Command cmd, const void* payload, Command resp, void* target);
};
#endif
