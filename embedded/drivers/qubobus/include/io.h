#include <stddef.h>
#include "qubobus.h"

#ifndef QUBOBUS_IO_H
#define QUBOBUS_IO_H

typedef size_t (*raw_io_function)(void*, size_t);
typedef void* (*malloc_function)(size_t);
typedef void (*free_function)(void*);

typedef struct _IO_State {
    /*
     * External functions required to implement io operations.
     */
    raw_io_function read_raw;
    raw_io_function write_raw;

    /*
     * Dynamic allocation library functions.
     */
    malloc_function malloc;
    free_function free;

    /*
     * State information for the connection itself.
     */
    uint16_t next_seq_num;

} IO_State;

typedef struct _Message {
    /* 
     * Details about the message structure itself
     * These 
     */
    struct Message_Header header;
    struct Message_Footer footer;

    /* 
     * Details about the contents of the message
     * These may be unintelligible depending on the message type
     */
    struct Error_Header error;
    struct Data_Header data;

    /* 
     * Details about the payload that was read from the message.
     * This includes the pointer to the payload and it's size,
     * as well as a flag for tracking whether the payload was dynamically allocated.
     */
    void *payload;
    uint16_t payload_size;
    uint16_t is_dynamic;

} Message;


/* 
 * Function to initialize the IO_State struct with needed data to start interacting across the bus.
 */
void initialize(
        IO_State *state, 
        raw_io_function read, 
        raw_io_function write, 
        malloc_function malloc, 
        free_function free, 
        uint16_t priority);

/*
 * Function to connect to the other device on the bus.
 */
int connect(IO_State *state);

/*
 * Function to create a message with given payload size.
 * If payload is NULL but the size is nonzero, a sufficient buffer will be allocated.
 */
Message create_message(IO_State *state, uint16_t message_type, void *payload, size_t payload_size);

/*
 * Function to destroy a Message object
 * This releases any dynamically allocated memory
 */
void destroy_message(IO_State *state, Message *message);

/*
 * Function to write a message to the data line.
 * This assembles the message based on the configuration of the message.
 * The message is modified to reflect the message that will be recieved on the other end.
 */
void send_message(IO_State *state, Message *message);

/*
 * Function to read an incoming message from the data bus.
 * Takes a void* to buffer memory to use for storing the read payload
 * If the buffer is insufficient (NULL or too small), it will attempt to allocate memory for the message.
 */
Message recieve_message(IO_State *state, void *buffer, size_t buffer_size);

#endif
