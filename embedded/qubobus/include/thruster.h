#include <stdint.h>

#ifndef QUBOBUS_THRUSTER_H
#define QUBOBUS_THRUSTER_H

/* Host -> QSCU message containing a throttle setting for a single thruster. */
#define D_ID_THRUSTER_SET 50
struct Thruster_Set {
    uint8_t thruster_id;

    /* Throttle setting for this thruster */
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

    /* Current draw for this thruster at this time. */
    float thruster_A;
};

/* Host -> QSCU a set of configuration parameters for the thruster system. */
#define D_ID_THRUSTER_CONFIG 53
struct Thruster_Config {
    /* The global gain setting for all thrusters. Scaling factor for all thruster output. */
    float thruster_gain;

    /* Maximum current to allow to a single thruster before sending an overcurrent error. */
    float thruster_high_A;
};

/* Error sent when a thruster is not connected. */
#define E_ID_THRUSTER_UNREACHABLE 50

/* Error sent when a thruster exceeds its current limit. */
#define E_ID_THRUSTER_OVERCURRENT 51

/* Error sent when a thruster has not recieved a Set command, and will automatically shut off. */
#define E_ID_THRUSTER_WATCHDOG 52

#endif
