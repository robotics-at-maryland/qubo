/******************************************************************************
 * Endpoint.cpp
 * Implementation of the QuboBus Endpoint.
 *
 * Copyright (C) 2016 Robotics at Maryland
 * Copyright (C) 2016 Greg Harris <gharris1727@gmail.com>
 * All rights reserved.
 ******************************************************************************/

// UNIX Serial includes
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdarg.h>

// Header include
#include "QSCU.h"

#if QUBOBUS_PROTOCOL_VERSION != 3
#error Update me with new message defs!
#endif

#include <stdio.h>

QSCU::QSCU(std::string deviceFile, speed_t baud) 
    : _deviceFile(deviceFile), _termBaud(baud), _deviceFD(-1), _timeout({10,0}), _state(initialize(this, QSCU::readRaw, QSCU::writeRaw, (uint16_t)10))
{ 
   
}

QSCU::~QSCU() { closeDevice(); }

void QSCU::openDevice() {
    struct termios termcfg;
    int modemcfg = 0, fd = -1;
    /* Open the serial port and store into a file descriptor.
     * O_RDWR allows for bi-directional I/O, and
     * O_NONBLOCK makes it so that read/write does not block.
     */
    fd = open(_deviceFile.c_str(), O_RDWR, O_NONBLOCK);
    // Check to see if the device exists.
    if (fd == -1)
        throw QSCUException("Device '"+_deviceFile+"' unavaliable.");
    // Read the config of the interface.
    if(// tcgetattr
       (fd, &termcfg)) 
        throw QSCUException("Unable to read terminal configuration.");

    // Set the baudrate for the terminal
    if(cfsetospeed(&termcfg, _termBaud))
        throw QSCUException("Unable to set terminal output speed.");
    if(cfsetispeed(&termcfg, _termBaud))
        throw QSCUException("Unable to set terminal intput speed.");

    // Set raw I/O rules to read and write the data purely.
    cfmakeraw(&termcfg);

    // Configure the read timeout (deciseconds)
    termcfg.c_cc[VTIME] = 0;
    // Configure the minimum number of chars for read.
    termcfg.c_cc[VMIN] = 1;

    // Push the configuration to the terminal NOW.
    if(tcsetattr(fd, TCSANOW, &termcfg))
        throw QSCUException("Unable to set terminal configuration.");

    // Pull in the modem configuration
    if(ioctl(fd, TIOCMGET, &modemcfg))
        throw QSCUException("Unable to read modem configuration.");
    // Enable Request to Send
    modemcfg |= TIOCM_RTS;
    // Push the modem config back to the modem.
    if(ioctl(fd, TIOCMSET, &modemcfg))
        throw QSCUException("Unable to set modem configuration.");

    // Successful hardware connection!
    // Needs to be set for the qscu protocol library to make the connection.
    _deviceFD = fd;

    // Prepare to begin communication with the device.
    if (connect(&_state)) {
        closeDevice();
        throw QSCUException("Unable to sychronize the remote connection!");
    }
}

bool QSCU::isOpen() {return _deviceFD >= 0;}

void QSCU::assertOpen() { if (!isOpen()) throw QSCUException("Device needs to be open!"); }

void QSCU::closeDevice() {
    if (isOpen()) 
        close(_deviceFD);
    _deviceFD = -1;
}

/******************************************************************************
 * Internal Functionality
 * All of the following functions are meant for internal-use only
 ******************************************************************************/

static ssize_t QSCU::readRaw(void* io_host, void* blob, ssize_t bytes_to_read)
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

static ssize_t QSCU::writeRaw(void* io_host, void* blob, ssize_t bytes_to_write)
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

void QSCU::sendMessage(Transaction *transaction, void *payload, void *response) {
    char buffer[QUBOBUS_MAX_PAYLOAD_LENGTH];
    bool completed = false;

    Message recieved_message, sent = create_request(transaction, payload);

    while (!completed) {
        bool recieved = false;

        write_message(&_state, &sent);

        while (!recieved) {
            read_message(&_state, &recieved_message, buffer);

            if (checksum_message(&recieved_message) != recieved_message.footer.checksum) {
                Message response = create_error(&eChecksum, NULL);
                write_message(&_state, &response);
            } else {
                recieved = true;
            }
        }

        if (recieved_message.header.message_type == MT_RESPONSE) {
            if (recieved_message.header.message_id != transaction->id || recieved_message.payload_size != transaction->response)
                throw new QSCUException("Malformed response payload!");
            /* Copy the read message back into the response buffer. */
            memcpy(response, buffer, transaction->response);
            completed = true;
        } else if (recieved_message.header.message_type == MT_ERROR) {
            if (recieved_message.header.message_id == eChecksum.id) {
                //The other side got a checksum error, retry sending.
            } else {
                // throw new QSCUException(str(recieved_message.payload, recieved_message.payload_size));
            }
        } else {
            //throw new QSCUException("Unexpected response: " + recieved_message.header.message_type + ":" + recieved.header.message_id);
        }
    }
}


//I'm like 90% sure we won't need these

// ssize_t serial_read(void *io_host, void *buffer, size_t size) {
//     QSCU *qscu = (QSCU*) io_host;
//     qscu->readRaw(buffer, size);
// }

// ssize_t serial_write(void *io_host, void *buffer, size_t size) {
//     QSCU *qscu = (QSCU*) io_host;
//     qscu->writeRaw(buffer, size);
// }


