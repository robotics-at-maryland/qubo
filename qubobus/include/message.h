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

    /* ID for messages sent requesting the QSCU to perform an action. */
    MT_REQUEST,

    /* ID for messages in response to a request message. */
    MT_RESPONSE,

    /* ID for errors encountered while completing a request. */
    MT_ERROR,

    /* Invalid for message IDs, bookkeeping for the maximum of message types. */
    MT_MAX,
};

#define IS_MESSAGE_TYPE(X) ((MT_NULL < (X)) && ((X) < MT_MAX))

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
     * This value should be one among the message types defined above.
     */
    uint8_t message_type;

    /*
     * ID for this message, used to attribute it to a subsystem.
     * This is one of any of the message ids defined across different subsystems
     */
    uint8_t message_id;

    /* 
     * Sequence number of this message.
     * This is used for tracking message ordering.
     */
    uint16_t sequence_number;
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
    uint16_t version;
};

#endif
