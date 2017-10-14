#include <io.h>

/* Local function definitions. */
static int read_announce(IO_State *state, Message *message);
static int safe_io(void *io_host, raw_io_function raw_io, void *data, size_t size);
static uint16_t crc16(uint16_t crc, const void *data, size_t bytes);
static void create_message(Message *message, uint8_t message_type, uint8_t message_id, void *payload, size_t payload_size);

/*
 * API functionality
 */
IO_State initialize(void *io_host, raw_io_function read_raw, raw_io_function write_raw, uint16_t priority) {
    IO_State state = {0};

    state.io_host = io_host;
    state.read_raw = read_raw;
    state.write_raw = write_raw;

    state.local_sequence_number = priority;
    state.remote_sequence_number = 0;

    return state;
}

int init_connect(IO_State *state, void *buffer) {
    Message our_announce, their_announce, protocol, response;
    int master, success;

    /*
     * ANNOUNCE THIS DEVICE
     */

    /* Send an announce message to the other client. */
    create_message(&our_announce, MT_ANNOUNCE, 0, NULL, 0);

    if (write_message(state, &our_announce)) {
        return -1;
    }
    /*
     * SYNCHRONIZE WITH OTHER DEVICE
     */

    /*
     * Read in the other client's announce message.

     * This serves to synchronize the message frame alignment of both clients.
     */
    if (read_announce(state, &their_announce)) {
        return -1;
    }

    /* Save the other client's sequence number */
    state->remote_sequence_number = their_announce.header.sequence_number;

    /*
     * NEGOTIATE PROTOCOL
     */

    /* The master client initiates the handshake with a protocol message. */
    struct Protocol_Info protocol_info = {QUBOBUS_PROTOCOL_VERSION};
    create_message(&protocol, MT_PROTOCOL, 0, &protocol_info, sizeof(struct Protocol_Info));

    if (write_message(state, &protocol)) {
        return -1;
    }

    /* Attempt to read a protocol message from the opposite client. */
    if (read_message(state, &response, buffer)) {
        return -1;
    }

    /* Check that we got a protocol message back from the other client. */
    success = (response.header.message_type == MT_PROTOCOL);

    return !success;
}

int wait_connect(IO_State *state, void *buffer) {
    Message our_announce, their_announce, protocol, response;
    int master, success;

    /*
     * SYNCHRONIZE WITH OTHER DEVICE
     */

    /*
     * Read in the other client's announce message.
     * This serves to synchronize the message frame alignment of both clients.
     */
    if (read_announce(state, &their_announce)) {
        return -1;
    }

    /*
     * ANNOUNCE THIS DEVICE
     */

    /* Send an announce message to the other client. */
    create_message(&our_announce, MT_ANNOUNCE, 0, NULL, 0);

    if (write_message(state, &our_announce)){
        return -1;
    }

    /* Save the other client's sequence number */
    state->remote_sequence_number = their_announce.header.sequence_number;

    /*
     * NEGOTIATE PROTOCOL
     */

    /* Attempt to read a protocol message from the opposite client. */
    if (read_message(state, &response, buffer)) {
        return -1;
    }

    /* Send a reply to confirm or deny the connection. */
    struct Protocol_Info *protocol_info = (struct Protocol_Info*) buffer;

    /* Check the protocol sent against our own version. */
    success = (response.header.message_type == MT_PROTOCOL
            && protocol_info->version == QUBOBUS_PROTOCOL_VERSION);

    /* If it didn't match, change the response from echo to error. */
    if (!success) {
        response = create_error(&eProtocol, NULL);
    }

    /* The slave client must respond to the master's protocol message. */
    if (write_message(state, &response)) {
        return -1;
    }


    return !success;
}


Message create_request(Transaction const *transaction, void *payload) {
    Message message;
    create_message(&message, MT_REQUEST, transaction->id, payload, transaction->request);
    return message;
}

Message create_response(Transaction const *transaction, void *payload) {
    Message message;
    create_message(&message, MT_RESPONSE, transaction->id, payload, transaction->response);
    return message;
}

Message create_error(Error const *error, void *payload) {
    Message message;
    create_message(&message, MT_ERROR, error->id, payload, error->size);
    return message;
}

Message create_keep_alive(){
  Message message;
  create_message(&message, MT_KEEPALIVE, 0, NULL, 0);
  return message;
}

int read_message(IO_State *state, Message *message, void *buffer) {
  struct Message_Header header;
  size_t payload_size;
  int rc = -1;

  /* Read in just the message header. */
  if (safe_io(state->io_host, state->read_raw, &header, sizeof(struct Message_Header))) {
	goto fail;
  }

  /* Compute the payload size by subtracting known message structure sizes. */
  payload_size = header.num_bytes
	- sizeof(struct Message_Header)
	- sizeof(struct Message_Footer);

  /* Allocate storage for the message */
  create_message(message, 0, 0, buffer, payload_size);

  /* Copy the message header into the message storage. */
  message->header = header;

  /* Read in the data payload. */
  if (safe_io(state->io_host, state->read_raw, message->payload, message->payload_size)) {
	goto fail;
  }

  /* Read in the message footer. */
  if (safe_io(state->io_host, state->read_raw, &(message->footer), sizeof(struct Message_Footer))) {
	goto fail;
  }

  state->remote_sequence_number++;

  rc = 0;

 fail:

  return rc;
}

int write_message(IO_State *state, Message *message) {

    /*
     * ASSEMBLE HEADER & FOOTER
     */

    /* Set the message size to include the core parts of the message. */
    message->header.num_bytes =
        sizeof(struct Message_Header) +
        sizeof(struct Message_Footer) +
        message->payload_size;

    /* Set the sequence number of the message from the state structure */
    message->header.sequence_number = ++state->local_sequence_number;

    /* Write the checksum into the message footer. */
    message->footer.checksum = checksum_message(message);

    /*
     * WRITE THE MESSAGE
     */

    /* Write the message to the bus. */
    return safe_io(state->io_host, state->write_raw, &(message->header), sizeof(struct Message_Header)) ||
        safe_io(state->io_host, state->write_raw, message->payload, message->payload_size) ||
        safe_io(state->io_host, state->write_raw, &(message->footer), sizeof(struct Message_Footer));

}

uint16_t checksum_message(Message *message) {
    uint16_t checksum = 0;

    /* Compute the checksum for the message header. */
    checksum = crc16(checksum, &(message->header), sizeof(struct Message_Header));

    /* Compute the checksum for the payload itself. */
    checksum = crc16(checksum, message->payload, message->payload_size);

    return checksum;
}

#define ANNOUNCE_SIZE sizeof(struct Message_Header) + sizeof(struct Message_Footer)

static int read_announce(IO_State *state, Message *message) {
    uint8_t buffer[ANNOUNCE_SIZE];
    struct Message_Header *header = (struct Message_Header*) buffer;
    struct Message_Footer *footer = (struct Message_Footer*) (buffer + sizeof(struct Message_Header));
    int rc = -1;

    /* Read an entire message worth of data */
    if(safe_io(state->io_host, state->read_raw, buffer + 1, ANNOUNCE_SIZE - 1)){
        goto fail;
    }
    do {
        int i;
        for (i = 1; i < ANNOUNCE_SIZE; i++) {
            buffer[i-1] = buffer[i];
        }
        if (safe_io(state->io_host, state->read_raw, buffer + ANNOUNCE_SIZE - 1, 1)) {
            goto fail;
        }
    } while (
            header->num_bytes != ANNOUNCE_SIZE ||
            header->message_type != MT_ANNOUNCE ||
            footer->checksum != crc16(0, header, sizeof(struct Message_Header))
            );

    message->header = *header;
    message->footer = *footer;
    message->payload = NULL;
    message->payload_size = 0;

    rc = 0;

fail:

    return rc;
}

#undef ANNOUNCE_SIZE

static int safe_io(void *io_host, raw_io_function raw_io, void *data, size_t size) {
    size_t bytes_transferred = 0;
    char *val = (char*) data;
    while (bytes_transferred != size) {
        ssize_t ret = raw_io(io_host, val + bytes_transferred, size - bytes_transferred);
        if (ret <= 0) {
            break;
        }
        bytes_transferred += ret;
    }

    return bytes_transferred != size;
}

static uint16_t crc16(uint16_t crc, const void* ptr, size_t bytes) {
    const uint8_t* data = (const uint8_t*) ptr;
    for (; bytes > 0; bytes--, data++)
        crc += *data;
    return crc;
}

static void create_message(Message *message, uint8_t message_type, uint8_t message_id, void *payload, size_t payload_size) {

    message->header.message_type = message_type;
    message->header.message_id = message_id;

    message->payload = payload;
    message->payload_size = payload_size;
}
