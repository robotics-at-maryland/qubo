#include <io.h>

static Message read_announce(IO_State *state);

void initialize(
        IO_State *state, 
        raw_io_function read_raw, 
        raw_io_function write_raw, 
        malloc_function malloc, 
        free_function free,
        uint16_t priority) {

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

    /* Send an announce message to the other client. */
    our_announce = create_message(MT_ANNOUNCE, NULL, 0);
    write_message(state, &our_announce);

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
    destroy_message(&our_announce);
    destroy_message(&their_announce);


    /* The master client initiates the handshake with a protocol message. */
    if (master) {

        protocol = create_message(MT_PROTOCOL, &protocol_info, sizeof(struct Protocol_Info));
        write_message(state, &protocol);
        destroy_message(&protocol);

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
            destroy_message(&response);
            response = create_message(MT_ERROR, NULL, 0);
            response.error.err_code = E_ID_PROTOCOL;
            response.error.cause_id = cause;
        }

        /* The slave client must respond to the master's protocol message. */
        write_message(state, &response);

    }

    destroy_message(&response);

    return !success;
}
