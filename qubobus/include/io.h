#include <unistd.h>
#include "qubobus.h"

#ifndef QUBOBUS_IO_H
#define QUBOBUS_IO_H
#define READ_TIMEOUT_MSEC 2000
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
IO_State initialize(void *io_host, raw_io_function read, raw_io_function write, uint16_t priority);

/*
 * Function to actively connect to the other device.
 */
int init_connect(IO_State *state, void *payload);

/*
 * Function to passively wait for the other device to connect.
 */
int wait_connect(IO_State *state, void *payload);

/*
 * Function to create transaction messages with a specified payload.
 */
Message create_request(Transaction const *transaction, void *payload);
Message create_response(Transaction const *transaction, void *payload);
Message create_error(Error const *error, void *payload);
Message create_keep_alive();

/*
 * Function to write a message to the data line.
 * This assembles the message based on the configuration of the message.
 * The message is modified to reflect the message that will be recieved on the other end.
 */
int write_message(IO_State *state, Message *message);

/*
 * Function to read an incoming message from the data bus.
 * Takes a void* to buffer memory to use for storing the read payload
 * This buffer should have sufficient size to recieve any message
 */
int read_message(IO_State *state, Message *message, void *buffer);

uint16_t checksum_message(Message *message);

#endif
