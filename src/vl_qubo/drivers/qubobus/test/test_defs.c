/*
 * Simple diagnostic program that prints out the lengths of various Qubobus messages.
 */

#include <qubobus.h>

#if QUBOBUS_PROTOCOL_VERSION != 3
#error Update me with new message defs!
#endif

#include <stdio.h>

int message(const char *name, const char *suffix, uint16_t type, size_t size);
int transact(Transaction const *t);
int error(Error const *t);

int main() { 
    int success = 1;
    /* Tests for protocol-wide messages */
    success &= message(
            "Announce Message", "",
            MT_ANNOUNCE, 
            EMPTY);
    success &= message(
            "Protocol Message", "",
            MT_PROTOCOL, 
            sizeof(struct Protocol_Info));
    success &= message(
            "Keepalive Message", "",
            MT_KEEPALIVE, 
            EMPTY);

    /* Tests involving the protocol error messages. */
    success &= error(&eProtocol);
    success &= error(&eChecksum);
    success &= error(&eSequence);
    success &= error(&eTimeout);

    /* Tests involving the embedded subsystem */
    success &= transact(&tEmbeddedStatus);
    success &= error(&eEmbeddedError);

    /* Tests for messages involving the safety subsystem */
    success &= transact(&tSafetyStatus);
    success &= transact(&tSafetySetSafe);
    success &= transact(&tSafetySetUnsafe);

    /* Tests for messages involving the battery subsystem */
    success &= transact(&tBatteryStatus);
    success &= transact(&tBatteryShutdown);
    success &= transact(&tBatteryMonitorEnable);
    success &= transact(&tBatteryMonitorDisable);
    success &= transact(&tBatteryMonitorSetConfig);
    success &= transact(&tBatteryMonitorGetConfig);
    success &= error(&eBatteryUnreachable);

    /* Tests involving the power subsystem */
    success &= transact(&tPowerStatus);
    success &= transact(&tPowerRailEnable);
    success &= transact(&tPowerRailDisable);
    success &= transact(&tPowerMonitorSetConfig);
    success &= transact(&tPowerMonitorGetConfig);
    success &= error(&ePowerUnreachable);

    /* Tests involving the thruster subsystem */
    success &= transact(&tThrusterSet);
    success &= transact(&tThrusterStatus);
    success &= transact(&tThrusterSetConfig);
    success &= transact(&tThrusterGetConfig);
    success &= transact(&tThrusterMonitorEnable);
    success &= transact(&tThrusterMonitorDisable);
    success &= transact(&tThrusterMonitorSetConfig);
    success &= transact(&tThrusterMonitorGetConfig);
    success &= error(&eThrusterUnreachable);

    /* Tests involving the pneumatics subsystem */
    success &= transact(&tPneumaticsSet);
    success &= error(&ePneumaticsUnreachable);

    /* Tests involving the depth sensor subsystem */
    success &= transact(&tDepthStatus);
    success &= transact(&tDepthMonitorEnable);
    success &= transact(&tDepthMonitorDisable);
    success &= transact(&tDepthMonitorSetConfig);
    success &= transact(&tDepthMonitorGetConfig);
    success &= error(&eDepthUnreachable);

    /* Tests involving the debug subsystem */
    success &= transact(&tDebugLogRead);
    success &= transact(&tDebugLogEnable);
    success &= transact(&tDebugLogDisable);
    success &= error(&eDebugLogError);

    if (success) {
        printf("No unrecoverable errors found!\n");
    }
    return !success;
}

/* Function indicating the definition of a message. */
int message(const char *name, const char *suffix, uint16_t type, size_t size) {
    size += sizeof(struct Message_Header);
    size += sizeof(struct Message_Footer);
    printf("%s %s: %lu\n", name, suffix, size);
    return 1;
}

/* Function indicating the declaration of a single data message type. */
int transact(Transaction const *t) {
    /* Table of data types that may be defined. */
    static int types[M_ID_OFFSET_MAX] = {0};

    /* If the type indicated is garbage, error out and dont keep checking the type. */
    if (!IS_MESSAGE_ID(t->id)) {
        printf("ERROR: IMPROPER DATA TYPE ID!\n");
        return 0;
    } 
    /* If there was a type definition collision, error out. */
    if (types[t->id]) {
        printf("ERROR: DATA TYPE ID REUSED!\n");
        return 0;
    }
    types[t->id] = 1;

    /* Defer to the main message definition checker for printing. */
    return message(t->name, "Request", MT_REQUEST, t->request) &&
        message(t->name, "Response", MT_RESPONSE, t->response);
}

/* Function indicating the declaration of a single error message type. */
int error(Error const *e) {
    /* Table of error types that may be defined. */
    static int types[M_ID_OFFSET_MAX] = {0};

    /* If the type indicated is garbage, error out and dont keep checking the type. */
    if (!IS_MESSAGE_ID(e->id)) {
        printf("ERROR: IMPROPER ERROR TYPE ID!\n");
        return 0;
    } 
    /* If there was a type definition collision, error out. */
    if (types[e->id]) {
        printf("ERROR: ERROR TYPE ID REUSED!\n");
        return 0;
    }
    types[e->id] = 1;

    /* Defer to the main message definition checker for printing. */
    return message(e->name, "", MT_ERROR, e->size);
}

