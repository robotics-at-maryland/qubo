#include <stdint.h>

#ifndef QUBOBUS_DEPTH_H
#define QUBOBUS_DEPTH_H

/* Host -> QSCU Message requesting status information from the depth system. */
#define D_ID_DEPTH_STATUS_REQUEST 70

/* QSCU -> Host Message with depth status information. */
#define D_ID_DEPTH_STATUS 71
struct Depth_Status {
    /* Current reading of the depth sensor */
    float depth_m;
};

/* Host -> QSCU Message setting configuration details of the depth subsystem. */
#define D_ID_DEPTH_CONFIG 72
struct Depth_Config {
    /* Maximum depth before warnings are generated */
    float depth_max;
};

/* Error sent when the sensor is unreachable. */
#define E_ID_DEPTH_UNREACHABLE 70

/* Error sent when the depth sensor reading exceeds the limit. */
#define E_ID_DEPTH_WARNING 71

#endif
