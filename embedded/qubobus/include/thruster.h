#include <stdint.h>

#ifndef QUBOBUS_THRUSTER_H
#define QUBOBUS_THRUSTER_H

/* Host -> QSCU message containing a throttle setting for a single thruster. */
#define D_ID_THRUSTER_SET 50
struct Thruster_Set {
    uint8_t thruster_id;
    float throttle;
};

/* Host -> QSCU message containing a request for a single thruster current reading. */
#define D_ID_THRUSTER_CURRENT_REQUEST 51
struct Thruster_Current_Request {
    uint8_t thruster_id;
};

/* QSCU - > Host message containing a single thruster's current measurement. */
#define D_ID_THRUSTER_CURRENT 52
struct Thruster_Current {
    uint8_t thruster_id;
    float thruster_mA;
};

/* Error message sent when a thruster is not connected. */
#define ERR_ID_THRUSTER_UNREACHABLE 50

#endif
