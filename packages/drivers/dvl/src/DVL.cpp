/******************************************************************************
 * DVL.cpp
 * DVL Device API implementation.
 *
 * Copyright (C) 2015 Robotics at Maryland
 * Copyright (C) 2015 Greg Harris <gharris1727@gmail.com>
 * All rights reserved.
 ******************************************************************************/

// UNIX Serial includes
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <endian.h>

// Header include
#include "../include/DVL.h"

#include "impl.cpp"

   DVL::DVL(std::string deviceFile, DVLSpeed speed) 
: _deviceFile(deviceFile), _termBaud(speed.baud), _deviceFD(-1), _timeout({1,0})
{ }

DVL::~DVL() { closeDevice(); }

void DVL::openDevice() {
   struct termios termcfg;
   int modemcfg = 0, fd = -1;
   /* Open the serial port and store into a file descriptor.
    * O_RDWR allows for bi-directional I/O, and
    * O_NONBLOCK makes it so that read/write does not block.
    */
   fd = open(_deviceFile.c_str(), O_RDWR, O_NONBLOCK);
   // Check to see if the device exists.
   if (fd == -1)
      throw DVLException("Device '"+_deviceFile+"' unavaliable.");
   // Read the config of the interface.
   if(tcgetattr(fd, &termcfg)) 
      throw DVLException("Unable to read terminal configuration.");

   // Set the baudrate for the terminal
   if(cfsetospeed(&termcfg, _termBaud))
      throw DVLException("Unable to set terminal output speed.");
   if(cfsetispeed(&termcfg, _termBaud))
      throw DVLException("Unable to set terminal intput speed.");

   // Set raw I/O rules to read and write the data purely.
   cfmakeraw(&termcfg);

   // Configure the read timeout (deciseconds)
   termcfg.c_cc[VTIME] = 0;
   // Configure the minimum number of chars for read.
   termcfg.c_cc[VMIN] = 1;

   // Push the configuration to the terminal NOW.
   if(tcsetattr(fd, TCSANOW, &termcfg))
      throw DVLException("Unable to set terminal configuration.");

   // Pull in the modem configuration
   if(ioctl(fd, TIOCMGET, &modemcfg))
      throw DVLException("Unable to read modem configuration.");
   // Enable Request to Send
   modemcfg |= TIOCM_RTS;
   // Push the modem config back to the modem.
   if(ioctl(fd, TIOCMSET, &modemcfg))
      throw DVLException("Unable to set modem configuration.");

   // Successful execution!
   _deviceFD = fd;
}

bool DVL::isOpen() {return _deviceFD >= 0;}

void DVL::assertOpen() { if (!isOpen()) throw DVLException("Device needs to be open!"); }

void DVL::closeDevice() {
   if (isOpen()) 
      close(_deviceFD);
   _deviceFD = -1;
}

/******************************************************************************
 * Internal Functionality
 * All of the following functions are meant for internal-use only
 ******************************************************************************/

DVL::checksum_t DVL::crc_xmodem_update (checksum_t crc, uint8_t data)
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

DVL::checksum_t DVL::crc16(checksum_t crc, uint8_t* data, bytecount_t bytes){
   for (; bytes > 0; bytes--, data++)
      crc = crc_xmodem_update(crc, *data);
   return crc;
}

int DVL::readRaw(void* blob, int bytes_to_read)
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

int DVL::writeRaw(void* blob, int bytes_to_write)
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

DVL::Message DVL::readMessage()
{
   // Storage for packet bytecount.
   bytecount_t total_size;
   // Storage for calculating the checksum.
   checksum_t remote_checksum, checksum = 0x0;
   // Output structure, storage for the read memory.
   Message message;

   // Read in the header of the datagram packet: UInt16.
   if (readRaw(&total_size, sizeof(bytecount_t)))
      throw DVLException("Unable to read bytecount of incoming packet.");
   // Add the total size to the checksum.
   checksum = crc16(checksum, (uint8_t*) &total_size, sizeof(bytecount_t));
   // Convert the total size from big-endian to host-endian.
   total_size = be16toh(total_size);
   // Do not include the checksum, frameid, or bytecount in the payload.
   message.payload_size = 
      total_size - sizeof(checksum_t) - sizeof(bytecount_t) - sizeof(frameid_t);

   // Create the memory in the Message struct.
   message.payload = std::make_shared<std::vector<char>>();
   message.payload->reserve(message.payload_size);

   // Read in the frameid: UInt8.
   if (readRaw(&message.id, sizeof(frameid_t)))
      throw DVLException("Unable to read frameid of incoming packet");
   // Add the message id to the checksum.
   checksum = crc16(checksum, (uint8_t*) &message.id, sizeof(frameid_t));
   // Read in the payload and spit it into the vector storage.
   if (readRaw(message.payload->data(), message.payload_size))
      throw DVLException("Unable to read payload of incoming packet");
   // Add the data read in to the checksum.
   checksum = crc16(checksum, (uint8_t*) message.payload->data(), message.payload_size);
   // Read the remote checksum that the device computed.
   if (readRaw(&remote_checksum, sizeof(checksum_t)))
      throw DVLException("Unable to read checksum of incoming packet");
   // Convert the checksum from big-endian to host-endian.
   remote_checksum = be16toh(remote_checksum);

   // Validate the remote checksum
   if (checksum != remote_checksum)
      throw DVLException("Incoming packet checksum invalid.");
   
   // Everything succeeded. The packet was read in properly.
   return message;
}

void DVL::writeMessage(Message message)
{
   // Calculate the total packet length.
   bytecount_t total_size = 
      sizeof(bytecount_t) + sizeof(frameid_t) + message.payload_size + sizeof(checksum_t);
   // Storage for the checksum.
   checksum_t checksum = 0x0;

   // Convert from host-endian to big-endian
   total_size = htobe16(total_size);

   // Compute the checksum from the packet data.
   checksum = crc16(checksum, (uint8_t*) &total_size, sizeof(bytecount_t));
   checksum = crc16(checksum, (uint8_t*) &message.id, sizeof(frameid_t));
   checksum = crc16(checksum, (uint8_t*) message.payload->data(), message.payload_size);
   // Convert from host-endian to big-endian
   checksum = htobe16(checksum);

   // Attempt to write the datagram to the serial port.
   if (writeRaw(&total_size, sizeof(bytecount_t)))
      throw DVLException("Unable to write bytecount.");
   // Attempt to write the frameid to the serial port.
   if (writeRaw(&message.id, sizeof(frameid_t)))
      throw DVLException("Unable to write frameid.");
   // Attempt to write the payload to the serial port.
   if (writeRaw(message.payload->data(), message.payload_size))
      throw DVLException("Unable to write payload.");
   // Attempt to write the checksum to the serial port.
   if (writeRaw(&checksum, sizeof(checksum_t)))
      throw DVLException("Unable to write checksum.");
}

DVL::Message DVL::createMessage(Command cmd, const void* payload)
{
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

DVL::Command DVL::inferCommand(Message message)
{
   throw DVLException("Unimplemented");
}

void DVL::readCommand(Command cmd, void* target)
{
   // Read until the message we receive is the one we want.
   Message message;
   do {
      message = readMessage();
   } while (cmd.name != inferCommand(message).name);

   // Copy the data to the target memory.
   if (target != NULL && !memcpy(target, message.payload->data(),message.payload_size))
      throw DVLException("Unable to copy the read command to the caller's memory.");
}

void DVL::writeCommand(Command cmd, const void* payload)
{
   writeMessage(createMessage(cmd, payload));
}

void DVL::sendCommand(Command send, const void* payload, Command resp, void* target)
{
   if ((target != NULL) && memset(target, 0, resp.payload_size) == NULL)
      throw DVLException("Unable to clear command response target memory.");
   writeCommand(send, payload);
   readCommand(resp, target);
}
