#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_DEPTH_H
#define QUBOBUS_DEPTH_H

enum {
    /* Command requesting a status message about the depth sensor. */
    D_ID_DEPTH_STATUS_REQUEST = M_ID_OFFSET_DEPTH,

    /* Response to a status request command. */
    D_ID_DEPTH_STATUS,

    /* Command to enable background monitoring of the depth sensor. */
    D_ID_DEPTH_MONITOR_ENABLE,

    /* Command to disable background monitoring of the depth sensor. */
    D_ID_DEPTH_MONITOR_DISABLE,

    /* Command to set the background monitoring configuration. */
    D_ID_DEPTH_MONITOR_CONFIG
};

enum {
    /* Error sent when the sensor is unreachable. */
    E_ID_DEPTH_UNREACHABLE = M_ID_OFFSET_DEPTH,

    /* Error sent when the depth sensor reading exceeds the limit. */
    E_ID_DEPTH_MONITOR_SOFT_WARNING,

    /* Error sent when the depth sensor reading exceeds hard limits */
    E_ID_DEPTH_MONITOR_HARD_WARNING
};

/* Message payload of details from the depth sensor */
struct Depth_Status {
    /* Current reading of the depth sensor */
    float depth_m;
};

/* Message payload with background monitoring configuration details. */
struct Depth_Monitor_Config {
    /* Maximum depth before warnings are generated */
    float soft_depth_max;
    
    /* Maximum depth before important warnings are generated */
    float hard_depth_max;
};

#endif
