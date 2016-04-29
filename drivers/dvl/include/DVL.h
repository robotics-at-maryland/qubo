#ifndef DVL_H
#define DVL_H

/*
 * DVL.h
 * Header file for DVL API.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris1727@gmail.com>
 * All rights reserved.
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
class DVLException : public std::runtime_error
{
    public:
        DVLException(std::string message)
            : runtime_error(message) {}
};

/**
 * DVL API class
 * Contains all the low-level I/O abstraction, and allows the user to communicate
 * with the DVL hardware at a high level. Connects to a unix serial/usb terminal
 * and speaks the DVL Binary protocol to send and recieve packets of data.
 */
class DVL
{
#include "types.h"
#include "statics.h"
#include "functions.h"
    public:
        /**
         * Constructor for a new DVL interface.
         * @param (std::string) unix device name
         * @param (DVLSpeed) Baudrate to use for connection.
         */
        DVL(std::string deviceFile, DVLSpeed speed);
        /** Destructor that cleans up and closes the device. */
        ~DVL();
        /** 
         * Opens the device and configures the I/O terminal. 
         * Requires dialout permissions to the device in _deviceFile.
         */
        void openDevice();
        /** 
         * Checks if the DVL is currently open and avaiable 
         * @return (bool) whether the DVL is avaliable for other API operations.
         */
        bool isOpen();
        /** Ensures that the DVL is open, throws an exception otherwise. */
        void assertOpen();
        /** Disconnectes from the device and closes the teriminal. */
        void closeDevice();
    private: // Internal functionality.
        /** Unix file name to connect to */
        std::string _deviceFile;
        /** Data rate to communicate with */
        speed_t _termBaud;
        /** Serial port for I/O with the DVL */
        int _deviceFD;
        /** Timeout (sec,usec) on read/write */
        struct timeval _timeout;

        /** Sends a pause to the DVL, triggering it to restart */
        void sendBreak();

        /** Checksum function to compute modulus 65535 CRC16s. */
        checksum_t crc16(checksum_t crc, const void* data, int bytes);
        /** Read bytes to a blob, return the bytes not read. */
        int readRaw(void* blob, int bytes_to_read);
        /** Write bytes from a blob, return the bytes not written. */
        int writeRaw(void* blob, int bytes_to_write);

        /** Helper function for readMessage() that reads in a PD0 formatted message. */
        Message readPD0();
        /** Helper function for readMessage() that reads in a PD4 formatted message. */
        Message readPD4();
        /** Helper function for readMessage() that reads in a PD5 formatted message. */
        Message readPD5();
        /** Helper function for readMessage() that reads in a PD6 formatted message. */
        Message readPD6();
        /** Helper function for readMessage() that reads in a plaintext formatted message. */
        Message readText(char first);
        /** Read an incoming message and format it for interpreting. */
        Message readMessage();

        /** Write a command with any free arguments in a varargs list */
        void writeFormatted(Command cmd, va_list argv);
        /** Write a command and its arguments to the device. */
        void writeCommand(Command cmd, ...);
        /** Write a command with variable args and read something back */
        Message sendCommand(Command cmd, ...);
};
#endif
