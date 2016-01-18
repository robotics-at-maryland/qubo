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
#include <stdarg.h>

// Header include
#include "../include/DVL.h"

#include "statics.cpp"
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

void DVL::sendBreak() {
   // Send a break in the data line for n deciseconds.
   // The DVL specs for more than 300ms, 
   // but it 'may respond to breaks shorter than this'
   // Here we will spec for 400ms of break time.
   ioctl(_deviceFD, TCSBRKP, 4);
}

DVL::checksum_t DVL::crc16(checksum_t crc, void* ptr, int bytes) {
   char* data = (char*) ptr;
   for (; bytes > 0; bytes--, data++)
      crc += *data;
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
   // Output structure, storage for the read memory.
   Message message;
   // Storage for calculating the checksum.
   checksum_t remote_checksum, checksum = 0x0;
   // Storage for the beginning of a message
   frameid_t start;
   // Bytecounts for perts of the PD0 data format.
   bytecount_t header_bytes = sizeof(PD0_Header);
   bytecount_t total_bytes;
   bytecount_t payload_bytes;
   // Temporary veriables for decoding the PD0 format.
   uint8_t data_types_i;
   data_offset_t *offsets;
   // Create the memory in the Message struct.
   message.payload = std::make_shared<std::vector<char>>();

   // Read in the header of the datagram packet: UInt16.
   if (readRaw(&start, sizeof(frameid_t)))
      throw DVLException("Unable to read beginning of incoming packet.");
   // Compute the local checksum with the read data.
   checksum = crc16(checksum, &start, sizeof(frameid_t));

   // From the first bytes, determine the type of message being sent in.
   switch (start) {
      case kPD0HeaderID: // We grabbed a PD0 packet that we have to read.
         message.format = FORMAT_PD0;
         message.payload->reserve(sizeof(PD0_Header));
         // Read in the header and spit it into the vector storage.
         if (readRaw(message.payload->data(), header_bytes))
            throw DVLException("Unable to read header of incoming packet");
         // Compute what's needed to read in the rest of the payload.
         message.pd0_header = (PD0_Header*) message.payload->data();
         total_bytes = message.pd0_header->bytes_in_ensemble;
         payload_bytes = total_bytes - sizeof(checksum_t) - sizeof(frameid_t);
         // Reserve enough storage for the payload.
         message.payload->reserve(payload_bytes);

         // Read in the remaining payload and spit it into the vector storage.
         if (readRaw(message.payload->data()+ header_bytes, payload_bytes - header_bytes))
            throw DVLException("Unable to read payload of incoming packet");
         // Compute the local checksum with the read data.
         checksum = crc16(checksum, message.payload->data(), payload_bytes);

         // Read the remote checksum that the device computed.
         if (readRaw(&remote_checksum, sizeof(checksum_t)))
            throw DVLException("Unable to read checksum of incoming packet");

         // Compare the checksums to validate the message.
         if (checksum != remote_checksum)
            throw DVLException("Remote and local checksum mismatch");

         message.pd0_header = (PD0_Header*) message.payload->data();
         offsets = (data_offset_t*) message.payload->data() + header_bytes;

         for(data_types_i = 0; data_types_i < message.pd0_header->data_types; data_types_i++) {
            data_offset_t offset = offsets[data_types_i];
            char* frame = message.payload->data() + offset;
            char* body = frame + sizeof(frameid_t);
            switch (*((frameid_t*) frame)) {
               case kPD0FixedLeaderID:
                  message.pd0_fixed =(PD0_FixedLeader*) body;
                  break;
               case kPD0VariableLeaderID:
                  message.pd0_variable = (PD0_VariableLeader*) body;
                  break;
               case kPD0VelocityDataID:
                  message.pd0_velocity = (PD0_CellShortFields*) body;
                  break;
               case kPD0CorrelationMagnitudeID:
                  message.pd0_correlation = (PD0_CellByteFields*) body;
                  break;
               case kPD0EchoIntensityID:
                  message.pd0_echo_intensity = (PD0_CellByteFields*) body;
                  break;
               case kPD0PercentGoodID:
                  message.pd0_percent_good = (PD0_CellByteFields*) body;
                  break;
               case kPD0StatusDataID:
                  message.pd0_status = (PD0_CellByteFields*) body;
                  break;
               case kPD0BottomTrackID:
                  message.pd0_bottom_track = (PD0_BottomTrack*) body;
                  break;
               case kPD0EnvironmentID:
                  message.pd0_environment = (PD0_Environment*) body;
                  break;
               case kPD0BottomTrackCommandID:
                  message.pd0_bottom_track_command = (PD0_BottomTrackCommand*) body;
                  break;
               case kPD0BottomTrackHighResID:
                  message.pd0_bottom_track_highres = (PD0_BottomTrackHighRes*) body;
                  break;
               case kPD0BottomTrackRangeID:
                  message.pd0_bottom_track_range = (PD0_BottomTrackRange*) body;
                  break;
               case kPD0SensorDataID:
                  message.pd0_sensor_data = (PD0_SensorData*) body;
                  break;
               default:
                  // This may be a sensor data packet, maybe we dont want to except.
                  throw DVLException("Unknown data format in PD0 packet");
            }
         }
         break;
      case kPD4HeaderID: // We grabbed a PD4 packet that we have to read (easy).
         message.format = FORMAT_PD4;
         message.payload->reserve(sizeof(PD4_Data));
         // Read in the payload and spit it into the vector storage.
         if (readRaw(message.payload->data(), sizeof(PD4_Data)))
            throw DVLException("Unable to read payload of incoming packet");
         // Compute the local checksum with the read data.
         checksum = crc16(checksum, message.payload->data(), sizeof(PD4_Data));
         // Read the remote checksum that the device computed.
         if (readRaw(&remote_checksum, sizeof(checksum_t)))
            throw DVLException("Unable to read checksum of incoming packet");
         // Compare the checksums to validate the message.
         if (checksum != remote_checksum)
            throw DVLException("Remote and local checksum mismatch");
         message.pd4_data = (PD4_Data*) message.payload->data();
         break;
      case kPD5HeaderID: // We grabbed a PD5 packet that we have to read (easy).
         message.format = FORMAT_PD5;
         message.payload->reserve(sizeof(PD5_Data));
         // Read in the payload and spit it into the vector storage.
         if (readRaw(message.payload->data(), sizeof(PD5_Data)))
            throw DVLException("Unable to read payload of incoming packet");
         // Compute the local checksum with the read data.
         checksum = crc16(checksum, message.payload->data(), sizeof(PD4_Data));
         // Read the remote checksum that the device computed.
         if (readRaw(&remote_checksum, sizeof(checksum_t)))
            throw DVLException("Unable to read checksum of incoming packet");
         // Compare the checksums to validate the message.
         if (checksum != remote_checksum)
            throw DVLException("Remote and local checksum mismatch");
         message.pd5_data = (PD5_Data*) message.payload->data();
         break;
      default:
         char text;
         message.format = FORMAT_TEXT;
         do {
            // Read in a single char and push it onto the payload.
            if (readRaw(&text, sizeof(char)))
               throw DVLException("Unable to read text character");
            // Disregard some characters from the input.
            if (text != '\r' && text != '>' && text != '<')
               message.payload->push_back(text);
            // This might all screw up if the command fails.
         } while (text != '>' && text != '<'); // Read until the next prompt appears.
         message.payload->push_back('\0'); // Null terminate the string
         message.text = message.payload->data();
         break;
   }
   // Everything succeeded. The packet was read in properly.
   return message;
}

#define BUF_SIZE 1024

void DVL::writeCommand(Command cmd, ...)
{
   char buffer[BUF_SIZE];
   char cr = '\r';
   va_list argv;
   // The second argument should be the last defined function arg. 
   va_start(argv,cmd);
   // Assemble the command to a string from the format string.
   int bytes = vsnprintf(buffer, BUF_SIZE, cmd.format, argv);
   // Bytes written will not include the null char at the end.
   if (bytes == BUF_SIZE - 1)
      throw DVLException("Write buffer overflow");
   // Write the command to the output line to the DVL.
   if (writeRaw(buffer, bytes))
      throw DVLException("Unable to send message");
   // Send a carriage return to tell the DVL input is finished.
   if (writeRaw(&cr, 1))
      throw DVLException("Unable to send carriage return");
}

DVL::Message DVL::sendCommand(Command cmd, ...)
{
    va_list argv;
    va_start(argv,cmd);
    writeCommand(cmd, argv);
    return readMessage();
}

