/**
 * Qubobus Message Definition File
 * These Message types make up the core of the Qubobus Protocol.
 *
 * Copyright (C) 2016 Robotics at Maryland
 * Copyright (C) 2016 Greg Harris <gharris1727@gmail.com>
 * All rights reserved.
 */

#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_MESSAGE_H
#define QUBOBUS_MESSAGE_H

/* 
 * Message type values to distinguish broad categories of messages.
 */
enum {
    /* INVALID ID for the lower limit of message types. */
    MT_NULL,
    /* ID for announce messages trying to synchronize the connection. */
    MT_ANNOUNCE,
    /* ID for messages with details of the protocol implementation. */
    MT_PROTOCOL,
    /* ID for message sent to monitor link state. */
    MT_KEEPALIVE,
    /* ID for messages about errors that have occurred. */
    MT_ERROR,
    /* ID for messages with data payloads. */
    MT_DATA,
    /* Invalid for message IDs, bookkeeping for the maximumof message types. */
    MT_MAX,
};

#define IS_MESSAGE_TYPE(X) ((MT_NULL < (X)) && ((X) < MT_MAX))

enum {
    /* Error sent when a protocol mismatch ocurrs. */
    E_ID_PROTOCOL = M_ID_OFFSET_CORE,

    /* Error sent when a checksum mismatch ocurrs */
    E_ID_CHECKSUM,

    /* Error sent after a long period of recieve inactivity. */
    E_ID_TIMEOUT
};

/**
 * Header for all messages.
 * This block should appear at the very beginning of every message.
 * It contains various information about the message and how it should be interpreted.
 */
struct Message_Header {
    /* 
     * Number of bytes in this message.
     * This total includes the whole header, the payload, and the footer.
     */
    uint16_t num_bytes;

    /* 
     * Type of this message.
     * This value should be one among the types #defined above.
     */
    uint16_t message_type;

    /* 
     * Sequence number of this message.
     * This is used for referencing in later messages. 
     */
    uint16_t seq_num;
};

/**
 * Footer for all messages.
 * This block should appear after the message payload, as the last bytes in any Message.
 */
struct Message_Footer {
    /* 
     * Checksum of all bytes previous in the message.
     * This does not include itself.
     */
    uint16_t checksum;
};

/**
 * Protocol Information Block.
 * Stores details about the protocol, and is sent in a Protocol Message.
 */
struct Protocol_Info {
    /* 
     * Version number of the protocol implemented by the client.
     */
    uint16_t version;
};

/**
 * Header of an Error Message.
 * This is expected to be at the very beginning of the payload in message that have MT_ERROR as the message_type.
 */
struct Error_Header {
    /* 
     * Error code of this Error Message.
     */
    uint16_t err_code;

    /* 
     * Reference to past message that may be the cause of this error. 
     * This may not be relevant depending on the error id.
     */
    uint16_t cause_id;
};

/**
 * Header of a Data Message.
 * This is expected to be at the very beginning of the payload in messages that have MT_DATA as the message_type.
 */
struct Data_Header {
    /* 
     * ID of the data message being sent. 
     * This determines how to interpret the rest of the data payload. 
     */
    uint16_t data_id;
};

#endif
