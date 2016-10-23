#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_SAFETY_H
#define QUBOBUS_SAFETY_H

enum {
    /* Host -> QSCU Command to request a Safety_Status message. */
    D_ID_SAFETY_STATUS_REQUEST = M_ID_OFFSET_SAFETY,

    /* QSCU -> Host Message with status information about the safety subsystem */
    D_ID_SAFETY_STATUS,

    /* Host -> QSCU Command to make the robot safe to handle. */
    D_ID_SAFETY_SET_SAFE,

    /* Host -> QSCU Command to make the robot unsafe */
    D_ID_SAFETY_SET_UNSAFE
};

/* Payload for a status message about the safety subsystem. */
struct Safety_Status {
    uint8_t hardware_sw;
    uint8_t sofware_sw;
};


#endif
