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

#include <stdio.h>

DVL::DVL(std::string deviceFile, DVLSpeed speed) 
    : _deviceFile(deviceFile), _termBaud(speed.baud), _deviceFD(-1), _timeout({10,0})
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

    // Prepare to begin communication with the device.
    sendBreak();
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
    // Clear any data that is in the read buffer so we start off with a clean slate.
    readMessage();
}

DVL::checksum_t DVL::crc16(checksum_t crc, const void* ptr, int bytes) {
    const char* data = (const char*) ptr;
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

DVL::Message DVL::readPD0()
{
    // Create a Message that we can return to the caller.
    Message message = {FORMAT_PD0, std::make_shared<Payload>()};
    // Single character buffer for dummy data.
    char dummy;
    // Create storage for the checksum calculations.
    checksum_t remote_checksum, checksum = 0x0;
    // Bytecounts for parts of the PD0 data format.
    bytecount_t header_bytes = sizeof(PD0_Header);
    bytecount_t total_bytes;
    bytecount_t payload_bytes;
    // Temporary veriables for decoding the PD0 format.
    uint8_t data_types_i;
    data_offset_t *offsets;

    // Read in the data source (dummy data) to prepare to read real data.
    if (readRaw(&dummy, sizeof(dummy)))
        throw DVLException("Unable to read data source of incoming packet");
    // Compute the local checksum of data already read in.
    checksum = crc16(checksum, &kPD0HeaderID, sizeof(kPD0HeaderID));

    // Prepare to read the header information
    message.payload->reserve(header_bytes);
    // Read in the header and spit it into the vector storage.
    if (readRaw(message.payload->data(), header_bytes))
        throw DVLException("Unable to read header of incoming packet");
    // Compute the local checksum with the read data.
    checksum = crc16(checksum, message.payload->data(), header_bytes);

    // Get a nice pointer to the header struct, this will get invalidated later.
    message.pd0_header = (PD0_Header*) message.payload->data();
    // Pull the total number of bytes from the header we just read.
    total_bytes = message.pd0_header->bytes_in_ensemble;
    // Calculate the number of bytes we have yet to read.
    payload_bytes = total_bytes - header_bytes - sizeof(kPD0HeaderID);
    // Reserve enough storage for the payload.
    message.payload->reserve(header_bytes + payload_bytes);
    // This invalidates the header we read in, so just to be safe null it out.
    message.pd0_header = NULL;

    // Read in the remaining payload and spit it into the vector storage.
    if (readRaw(message.payload->data() + header_bytes, payload_bytes))
        throw DVLException("Unable to read payload of incoming packet");
    // Compute the local checksum with the read data.
    checksum = crc16(checksum, message.payload->data() + header_bytes, payload_bytes);

    // Read the remote checksum that the device computed.
    if (readRaw(&remote_checksum, sizeof(remote_checksum)))
        throw DVLException("Unable to read checksum of incoming packet");

    // Compare the LSBs of the checksums to validate the message.
    if ((remote_checksum ^ checksum) & 0xff)
        throw DVLException("Remote and local checksum mismatch");

    // Reassign the header pointer, because the old pointer got invalidated by reserve()
    message.pd0_header = (PD0_Header*) message.payload->data();
    // Get a pointer to the data offset array.
    offsets = (data_offset_t*) message.payload->data() + header_bytes;
    // Loop over the offset array to find all the data in the packet.
    for(data_types_i = 0; data_types_i < message.pd0_header->data_types; data_types_i++) {
        // Pull out the offset from the offset array.
        data_offset_t offset = offsets[data_types_i];
        // Get a pointer to the beginning of the data referenced by the offset.
        char* frame = message.payload->data() + offset - sizeof(kPD0HeaderID);
        // The body of the data at that point is one frameid_t length farther.
        char* body = frame + sizeof(frameid_t);
        // Access the frameid at the start of the frame and decide what is there.
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
                // If the packet is well formed then this shouldnt happen.
                printf("Unrecogized data header: 0x%x\n",*((frameid_t*) frame));
                //throw DVLException("Unknown data format in PD0 packet");
        }
    }
    // Return the finished message to the caller.
    return message;
}

DVL::Message DVL::readPD4()
{
    // Create a Message that we can return to the caller.
    Message message = {FORMAT_PD4, std::make_shared<Payload>()};
    // Create storage for the checksum operation.
    checksum_t remote_checksum, checksum = 0x0;
    // Compute the checksum with the read info so far.
    checksum = crc16(checksum, &kPD4HeaderID, sizeof(kPD4HeaderID));
    // Pre-allocate storage for the whole packet.
    message.payload->reserve(sizeof(PD4_Data));
    // Read in the payload and spit it into the vector storage.
    if (readRaw(message.payload->data(), sizeof(PD4_Data)))
        throw DVLException("Unable to read payload of incoming packet");
    // Compute the local checksum with the read data.
    checksum = crc16(checksum, message.payload->data(), sizeof(PD4_Data));
    // Read the remote checksum that the device computed.
    if (readRaw(&remote_checksum, sizeof(remote_checksum)))
        throw DVLException("Unable to read checksum of incoming packet");

    // Compare the checksums to validate the message.
    if ((remote_checksum ^ checksum) & 0xff)
        throw DVLException("Remote and local checksum mismatch");
    // Store the pointer to allow easy access to the data.
    message.pd4_data = (PD4_Data*) message.payload->data();
    // Return the finished message to the caller.
    return message;
}

DVL::Message DVL::readPD5()
{
    // Create a Message that we can return to the caller.
    Message message = {FORMAT_PD5, std::make_shared<Payload>()};
    // Create storage for the checksum operation.
    checksum_t remote_checksum, checksum = 0x0;
    // Pre-allocate storage for the whole packet.
    message.payload->reserve(sizeof(PD5_Data));
    // Compute the checksum with the read info so far.
    checksum = crc16(checksum, &kPD5HeaderID, sizeof(kPD5HeaderID));
    // Read in the payload and spit it into the vector storage.
    if (readRaw(message.payload->data(), sizeof(PD5_Data)))
        throw DVLException("Unable to read payload of incoming packet");
    // Compute the local checksum with the read data.
    checksum = crc16(checksum, message.payload->data(), sizeof(PD5_Data));
    // Read the remote checksum that the device computed.
    if (readRaw(&remote_checksum, sizeof(remote_checksum)))
        throw DVLException("Unable to read checksum of incoming packet");

    // Compare the checksums to validate the message.
    if ((remote_checksum ^ checksum) & 0xff)
        throw DVLException("Remote and local checksum mismatch");
    // Store the pointer to allow easy access to the data.
    message.pd5_data = (PD5_Data*) message.payload->data();
    // Return the finished message to the caller.
    return message;
}

DVL::Message DVL::readPD6() 
{
    // Create a Message that we can return to the caller.
    Message message = {FORMAT_PD6, std::make_shared<Payload>()};
    // Create buffer storage to read in data char-by-char.
    char text;
    // Keep a list of the starts of strings.
    int attitude = 0, timing = 0, w_instrument, b_instrument;
    int w_ship, b_ship, w_earth, b_earth, w_distance, b_distance;
    // Track the beginning of each string offset in the payload
    int start = 0;
    // Number of lines is dynamic, so keep track of what we've already seen.
    bool water = false, bottom = false;
    // Put the character we already read in the vector.
    message.payload->push_back(':');
    // Read in the data line-by-line and parse
    // We need to read at least 3 lines, and then the limit changes based on what
    // lines we have seen so far. Seeing a ":W" line or ":B" line adds 4 to the total.
    for (int lines = 0; lines < 3 || lines < (2 + (water?4:0) + (bottom?4:0)); lines++) {
        do {
            // Read in a single char and push it onto the payload.
            if (readRaw(&text, sizeof(char)))
                throw DVLException("Unable to read text character");
            // Disregard some characters from the input.
            if (text != '\r' && text != '\n')
                message.payload->push_back(text);
            // Read until the end of the line.
        } while (text != '\n');
        // Null terminate the string
        message.payload->push_back('\0');

        // If we managed to read at least three chars (:XX) then take a look at what we read.
        if ((message.payload->size() - start) < 3)
            throw new DVLException("Line was too short to interpret.");
        // Look at the first command char.
        switch(message.payload->at(start + 1)) {
            case 'S': // System Attitude Data
                attitude = start;
                break;
            case 'T': // Time/Scaling Data
                timing = start;
                break;
            case 'B': // Bottom Track Data
                bottom = true;
                switch(message.payload->at(start + 2)) {
                    case 'I':
                        b_instrument = start;
                        break;
                    case 'S':
                        b_ship = start;
                        break;
                    case 'E':
                        b_earth = start;
                        break;
                    case 'D':
                        b_distance = start;
                        break;
                }
                break;
            case 'W': // Water Mass Data
                water = true;
                switch(message.payload->at(start + 2)) {
                    case 'I':
                        w_instrument = start;
                        break;
                    case 'S':
                        w_ship = start;
                        break;
                    case 'E':
                        w_earth = start;
                        break;
                    case 'D':
                        w_distance = start;
                        break;
                }
                break;
            default:
                throw DVLException("Unknown PD6 line format char.");
        }
        start = message.payload->size();
    }
    message.pd6_attitude = attitude + message.payload->data();
    message.pd6_timing = timing + message.payload->data();
    if (water) {
        message.pd6_w_instrument = w_instrument + message.payload->data();
        message.pd6_w_ship = w_ship + message.payload->data();
        message.pd6_w_earth = w_earth + message.payload->data();
        message.pd6_w_distance = w_distance + message.payload->data();
    }
    if (bottom) {
        message.pd6_b_instrument = b_instrument + message.payload->data();
        message.pd6_b_ship = b_ship + message.payload->data();
        message.pd6_b_earth = b_earth + message.payload->data();
        message.pd6_b_distance = b_distance + message.payload->data();
    }
    // Return the finished message to the caller.
    return message;
}

DVL::Message DVL::readText(char text)
{
    // Create a Message that we can return to the caller.
    Message message = {FORMAT_TEXT, std::make_shared<Payload>()};
    // Read until the next prompt appears.
    while (text != '>') {
        // Put what we just read into storage.
        message.payload->push_back(text);
        // Read in a single char and push it onto the payload.
        if (readRaw(&text, sizeof(text)))
            throw DVLException("Unable to read text character");
    }
    // Null terminate the string
    message.payload->push_back('\0');
    // The string pointer will start at the beginning of the payload vector.
    message.text = message.payload->data();
    // Return the finished message to the caller.
    return message;
}

DVL::Message DVL::readMessage()
{
    char first = 0;
    // Read in the beginning of the incoming message.
    do {
        if (readRaw(&first, sizeof(first)))
            throw DVLException("Unable to read beginning of incoming message.");
    } while (!first);
    // From the first byte, determine the type of message being sent in.
    switch (first) {
        case kPD0HeaderID & 0xff: // We grabbed a PD0 packet that we have to read.
            return readPD0();
        case kPD4HeaderID & 0xff: // We grabbed a PD4/5 packet that we have to read.
            if (readRaw(&first, sizeof(first)))
                throw DVLException("Unable to read beginning of PD4/5 message.");
            if (first) {
                return readPD5();
             } else {
                return readPD4();
            }
        case ':': // We grabbed a PD6 packet that we have to read.
            return readPD6();
        case '>': // The message that came back was just a prompt.
            return {FORMAT_EMPTY};
        default:
            return readText(first);
    }
}

#define BUF_SIZE 1024

void DVL::writeFormatted(Command cmd, va_list argv)
{
    char buffer[BUF_SIZE];
    char cr = '\r';
    // Assemble the command to a string from the format string.
    int bytes = vsnprintf(buffer, BUF_SIZE, cmd.format, argv);
    // Bytes written will not include the null char at the end.
    if (bytes == BUF_SIZE - 1)
        throw DVLException("Write buffer overflow");
    // Write the command to the output line to the DVL.
    if (writeRaw(buffer, bytes))
        throw DVLException("Unable to send command");
    // Write a carriage return to complete the command line.
    if (writeRaw(&cr, 1))
        throw DVLException("Unable to send carriage return");
    // Read the echoed command that comes back from the DVL.
    if (readRaw(buffer, bytes))
        throw DVLException("Unable to read command");
    // Read the echoed carriage return.
    if (readRaw(&cr, 1))
        throw DVLException("Unable to read carriage return");
    // Read the linefeed that the DVL responds with.
    if (readRaw(&cr, 1))
        throw DVLException("Unable to read linefeed");
}

void DVL::writeCommand(Command cmd, ...)
{
    va_list argv;
    va_start(argv, cmd);
    assertOpen();
    writeFormatted(cmd, argv);
    va_end(argv);
}

DVL::Message DVL::sendCommand(Command cmd, ...)
{
    va_list argv;
    va_start(argv, cmd);
    assertOpen();
    writeFormatted(cmd, argv);
    va_end(argv);
    return readMessage();
}

