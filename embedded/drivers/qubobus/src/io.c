#include <io.h>

/* Local function definitions. */
static Message read_announce(IO_State *state);
static void safe_io(raw_io_function raw_io, void *data, size_t size);
static uint16_t crc16(uint16_t crc, const void *data, size_t bytes);
static uint16_t checksum_message(Message *message);
static Message read_message(IO_State *state, void *buffer, size_t buffer_size);
static void write_message(IO_State *state, Message *message);

/*
 * API functionality
 */
void initialize(IO_State *state, raw_io_function read_raw, raw_io_function write_raw, malloc_function malloc, free_function free, uint16_t priority) {

    state->read_raw = read_raw;
    state->write_raw = write_raw;
    state->malloc = malloc;
    state->free = free;
    state->next_seq_num = priority;

}

int connect(IO_State *state) {
    Message our_announce;
    Message their_announce;
    Message protocol;
    Message response;
    struct Protocol_Info protocol_info = {QUBOBUS_PROTOCOL_VERSION};
    int master, success;

    /*
     * ANNOUNCE THIS DEVICE
     */

    /* Send an announce message to the other client. */
    our_announce = create_message(state, MT_ANNOUNCE, NULL, 0);
    write_message(state, &our_announce);

    /*
     * SYNCHRONIZE WITH OTHER DEVICE
     */

    /* 
     * Read in the other client's announce message. 
     * This serves to synchronize the message frame alignment of both clients.
     */
    their_announce = read_announce(state);

    /* 
     * In order to generate asymmetry, we compare the announce sequence numbers
     * If ours is lower, then we are the ones in control of the handshake.
     */
    master = (our_announce.header.sequence_number < their_announce.header.sequence_number);

    /* Clients agree to use the non-master sequence number */
    if (master) {

        state->next_seq_num = their_announce.header.sequence_number;

    }

    /* Destroy the announce messages; we dont need them any more. */
    destroy_message(state, &our_announce);
    destroy_message(state, &their_announce);

    /*
     * NEGOCIATE PROTOCOL
     */

    /* The master client initiates the handshake with a protocol message. */
    if (master) {

        protocol = create_message(state, MT_PROTOCOL, &protocol_info, sizeof(struct Protocol_Info));
        write_message(state, &protocol);
        destroy_message(state, &protocol);

    }

    /* Attempt to read a protocol message from the opposite client. */
    response = read_message(state, &protocol_info, sizeof(struct Protocol_Info));

    /* Handle the response from the opposite client. */
    if (master) {

        /* Check that we got a protocol message back from the other client. */
        success = (response.header.message_type == MT_PROTOCOL);

    } else {

        /* Check the protocol sent against our own version. */
        success = (protocol_info.version == QUBOBUS_PROTOCOL_VERSION);

        /* If it didn't match, change the response from echo to error. */
        if (!success) { 
            uint16_t cause = response.header.sequence_number;
            destroy_message(state, &response);
            response = create_message(state, MT_ERROR, NULL, 0);
            response.error.err_code = E_ID_PROTOCOL;
            response.error.cause_id = cause;
        }

        /* The slave client must respond to the master's protocol message. */
        write_message(state, &response);

    }

    destroy_message(state, &response);

    return !success;
}

Message create_message(IO_State *state, uint16_t message_type, void *payload, size_t payload_size) {

    /* Initialize the message to be empty. */
    Message message = {0};

    /* Set the message type in the header of the message. */
    message.header.message_type = message_type;

    /* If we need space for a payload, we need to make sure there is a sufficient buffer. */
    if (payload_size > 0) {

        /* If we didn't specify a buffer, then allocate one to use. */
        if (!payload) {
            payload = state->malloc(payload_size);

            /* Tag the message as having dynamically allocated memory. */
            message.is_dynamic = 1;
        }

        /* Set the relevant payload details */
        message.payload = payload;
        message.payload_size = payload_size;
    }

    /* Return the configured message to the caller. */
    return message;
}

void destroy_message(IO_State *state, Message *message) {
    /* If the message includes a dynamic buffer, make sure to free it. */
    if (message->is_dynamic) {
        state->free(message->payload);
    }

    /* Clear the message entirely back to 0 to avoid re-destroying it. */
    *message = (Message) {0};
}

void send_message(IO_State *state, Message *message) {
    write_message(state, message);
}

Message recieve_message(IO_State *state, void *buffer, size_t buffer_size) {

    Message message = read_message(state, buffer, buffer_size);

    if ((message.header.sequence_number + 1) != state->next_seq_num) {
        Message response = create_message(state, MT_ERROR, NULL, 0);
        response.error.err_code = E_ID_SEQUENCE;
        response.error.cause_id = message.header.sequence_number;
        write_message(state, &response);
        destroy_message(state, &response);
    }

    if (message.footer.checksum != checksum_message(&message)) {
        Message response = create_message(state, MT_ERROR, NULL, 0);
        response.error.err_code = E_ID_CHECKSUM;
        response.error.cause_id = message.header.sequence_number;
        write_message(state, &response);
        destroy_message(state, &response);
    }

    return message;
}

#define ANNOUNCE_SIZE sizeof(struct Message_Header) + sizeof(struct Message_Footer)

static Message read_announce(IO_State *state) {
    Message message = {0};
    uint8_t buffer[ANNOUNCE_SIZE];
    struct Message_Header *header = (struct Message_Header*) buffer;
    struct Message_Footer *footer = (struct Message_Footer*) (buffer + sizeof(struct Message_Header));

    /* Read an entire message worth of data */
    safe_io(state->read_raw, buffer + 1, ANNOUNCE_SIZE - 1);
    do {
        int i;
        for (i = 1; i < ANNOUNCE_SIZE; i++) {
            buffer[i-1] = buffer[i];
        }
        safe_io(state->read_raw, buffer + ANNOUNCE_SIZE - 1, 1);
    } while (
            header->num_bytes != ANNOUNCE_SIZE ||
            header->message_type != MT_ANNOUNCE ||
            footer->checksum != crc16(0, header, sizeof(struct Message_Header))
            );

    message.header = *header;
    message.footer = *footer;

    return message;
}

#undef ANNOUNCE_SIZE

static void safe_io(raw_io_function raw_io, void *data, size_t size) {
    size_t bytes_transferred = 0;
    while (bytes_transferred != size) {
        ssize_t ret = raw_io(data + bytes_transferred, size - bytes_transferred);
        if (ret < 0) {
            break;
        }
        bytes_transferred += ret;
    }
}

static uint16_t crc16(uint16_t crc, const void* ptr, size_t bytes) {
    const char* data = (const char*) ptr;
    for (; bytes > 0; bytes--, data++)
        crc += *data;
    return crc;
}

static uint16_t checksum_message(Message *message) {
    uint16_t checksum = 0;

    /* Compute the checksum for the message header. */
    checksum = crc16(checksum, &(message->header), sizeof(struct Message_Header));

    /* Compute the checksum for the message payload header. */
    switch (message->header.message_type) {
        case MT_ERROR:
            checksum = crc16(checksum, &(message->error), sizeof(struct Error_Header));
            break;
        case MT_DATA:
            checksum = crc16(checksum, &(message->data), sizeof(struct Data_Header));
            break;
    }

    /* Compute the checksum for the payload itself. */
    checksum = crc16(checksum, message->payload, message->payload_size);

    return checksum;
}

static Message read_message(IO_State *state, void *buffer, size_t buffer_size) {
    Message message;
    struct Message_Header header;
    size_t payload_size;

    /* Read in just the message header. */
    safe_io(state->read_raw, &header, sizeof(struct Message_Header));

    /* Compute the payload size by subtracting known message structure sizes. */
    payload_size = header.num_bytes
        - sizeof(struct Message_Header)
        - sizeof(struct Message_Footer);

    /* Subtract the payload header sizes. */
    switch (message.header.message_type) {
        case MT_ERROR:
            payload_size -= sizeof(struct Error_Header);
            break;
        case MT_DATA:
            payload_size -= sizeof(struct Data_Header);
            break;
    }

    /* Allocate storage for the message */
    message = create_message(
            state, 
            header.message_type, 
            buffer_size >= payload_size ? buffer : NULL,
            payload_size);

    /* Copy the message header into the message storage. */
    message.header = header;

    /* Read in the payload header. */
    switch (message.header.message_type) {
        case MT_ERROR:
            safe_io(state->read_raw, &(message.error), sizeof(struct Error_Header));
            break;
        case MT_DATA:
            safe_io(state->read_raw, &(message.data), sizeof(struct Data_Header));
            break;
    }

    /* Read in the data payload. */
    safe_io(state->read_raw, message.payload, message.payload_size);

    /* Read in the message footer. */
    safe_io(state->read_raw, &(message.footer), sizeof(struct Message_Footer));

    state->next_seq_num++;
}

static void write_message(IO_State *state, Message *message) {

    /*
     * ASSEMBLE HEADER & FOOTER
     */

    /* Set the message size to include the core parts of the message. */
    message->header.num_bytes = 
        sizeof(struct Message_Header) +
        sizeof(struct Message_Footer) +
        message->payload_size;


    /* Add message-type dependent data sizes. */
    switch (message->header.message_type) {
        case MT_ERROR:
            message->header.num_bytes += sizeof(struct Error_Header);
            break;
        case MT_DATA:
            message->header.num_bytes += sizeof(struct Data_Header);
            break;
    }

    /* Set the sequence number of the message from the state structure */
    message->header.sequence_number = state->next_seq_num++;

    /* Write the checksum into the message footer. */
    message->footer.checksum = checksum_message(message);

    /*
     * WRITE THE MESSAGE
     */

    /* Write the message header to the bus. */
    safe_io(state->write_raw, &(message->header), sizeof(struct Message_Header));

    /* Write the message payload header. */
    switch (message->header.message_type) {
        case MT_ERROR:
            safe_io(state->write_raw, &(message->error), sizeof(struct Error_Header));
            break;
        case MT_DATA:
            safe_io(state->write_raw, &(message->data), sizeof(struct Data_Header));
            break;
    }

    /* Write the data payload */
    safe_io(state->write_raw, message->payload, message->payload_size);

    /* Write the data footer. */
    safe_io(state->write_raw, &(message->footer), sizeof(struct Message_Footer));

}








