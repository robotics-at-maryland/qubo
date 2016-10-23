#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_THRUSTER_H
#define QUBOBUS_THRUSTER_H

enum {
    /* Host -> QSCU message containing a throttle setting for a single thruster. */
    D_ID_THRUSTER_SET = M_ID_OFFSET_THRUSTER,

    /* Host -> QSCU message containing a request for a single thruster current reading. */
    D_ID_THRUSTER_STATUS_REQUEST,

    /* QSCU - > Host message containing a single thruster's current measurement. */
    D_ID_THRUSTER_STATUS,

    /* Host -> QSCU a set of configuration parameters for the thruster system. */
    D_ID_THRUSTER_CONFIG,

    /* Host -> QSCU Command to enable background monitoring */
    D_ID_THRUSTER_MONITOR_ENABLE,
    
    /* Host -> QSCU Command to disable background monitoring */
    D_ID_THRUSTER_MONITOR_DISABLE,

    /* Host -> QSCU Command to set the thruster monitor configuration. */
    D_ID_THRUSTER_MONITOR_CONFIG,
};

enum {
    /* Error sent when a thruster is not connected. */
    E_ID_THRUSTER_UNREACHABLE = M_ID_OFFSET_THRUSTER,

    /* Error sent when a thruster has not recieved a Set command, and will automatically shut off. */
    E_ID_THRUSTER_TIMEOUT,

    /* Error sent when a thruster exceeds a soft limit. */
    E_ID_THRUSTER_MONITOR_SOFT_WARNING,

    /* Error sent when a thruster exceeds a hard limit */
    E_ID_THRUSTER_MONITOR_HARD_WARNING,
};

/* Payload to change the throttle setting of a single thruster. */
struct Thruster_Set {
    float throttle;
    uint8_t thruster_id;
};

/* Message requesting the status of a single thruster */
struct Thruster_Status_Request {
    uint8_t thruster_id;
};

/* Response payload containing thruster status measurements */
struct Thruster_Status {
    float thruster_A;
    float thruster_V;
    float throttle_setting;
    int16_t pwm_duty_cycle;
};

/* Payload of global thruster configuration details */
struct Thruster_Config {
    /* The global gain setting for all thrusters. Scaling factor for all thruster output. */
    float thruster_gain;
};

/* Payload of thruster background monitor configuration details. */
struct Thruster_Monitor_Config {
    /* Soft limits for parameters for the thrusters */
    float soft_thruster_high_A;
    float soft_thruster_low_V;

    /* Hard limits for thruster measurements. */
    float hard_thruster_high_A;
    float hard_thruster_low_V;
};

#endif
