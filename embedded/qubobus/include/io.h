#include <unistd.h>
#include "qubobus.h"

#ifndef QUBOBUS_IO_H
#define QUBOBUS_IO_H


//The read_raw and write_raw functions must match the spec for posix read/write
//for documentation on these type "man read" and "man right" into your terminal, those
//are the reads and writes we are talking about.

//The difference is that io_host pointer. This will be the first argument passed to your read/write functions
//it doesn't need to be a file descriptor or anything, it is meant to tell your read/write functions what to
//read/write to though.

typedef ssize_t (*raw_io_function)(void*, void*, size_t);

typedef struct _IO_State {
    /*
     * External functions required to implement io operations.
     */
    void *io_host;
    raw_io_function read_raw;
    raw_io_function write_raw;

    /*
     * State information for the connection itself.
     */
    uint16_t local_sequence_number;
    uint16_t remote_sequence_number;

} IO_State;

typedef struct _Message {
    /* 
     * Details about the message structure itself
     * These 
     */
    struct Message_Header header;
    struct Message_Footer footer;

    /* 
     * Details about the payload that was read from the message.
     * This includes the pointer to the payload and it's size,
     * as well as a flag for tracking whether the payload was dynamically allocated.
     */
    void *payload;
    uint16_t payload_size;

} Message;


/* 
 * Function to initialize the IO_State struct with needed data to start interacting across the bus.
 */

//The read_raw and write_raw functions must match the spec for posix read/write
//for documentation on these type "man read" and "man right" into your terminal, those
//are the reads and writes we are talking about.

//The difference is that io_host pointer. This will be the first argument passed to your read/write functions
//it doesn't need to be a file descriptor or anything, it is meant to tell your read/write functions what to
//read/write to though.

IO_State initialize(void *io_host, raw_io_function read, raw_io_function write, uint16_t priority);

/*
 * Function to connect to the other device on the bus.
 */
int connect(IO_State *state);

/*
 * Function to create transaction messages with a specified payload.
 */
Message create_request(Transaction const *transaction, void *payload);
Message create_response(Transaction const *transaction, void *payload);
Message create_error(Error const *error, void *payload);

/*
 * Function to write a message to the data line.
 * This assembles the message based on the configuration of the message.
 * The message is modified to reflect the message that will be recieved on the other end.
 */
void write_message(IO_State *state, Message *message);

/*
 * Function to read an incoming message from the data bus.
 * Takes a void* to buffer memory to use for storing the read payload
 * This buffer should have sufficient size to recieve any message
 */
void read_message(IO_State *state, Message *message, void *buffer);

uint16_t checksum_message(Message *message);

#endif
