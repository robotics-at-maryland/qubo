#include <stdint.h>

#ifndef QUBOBUS_DEPTH_H
#define QUBOBUS_DEPTH_H

/* Host -> QSCU A request for a depth reading. */
#define D_ID_DEPTH_REQUEST 70

/* QSCU -> Host A reading from the depth sensor */
#define D_ID_DEPTH 71
struct Depth_Reading {
    float depth_m;
};

/* Error generated when the sensor is unreachable. */
#define ERR_ID_DEPTH_UNREACHABLE 70

#endif
