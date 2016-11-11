#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_EMBEDDED_H
#define QUBOBUS_EMBEDDED_H

enum {
    M_ID_EMBEDDED_STATUS = M_ID_OFFSET_EMBEDDED,
};

enum {
    E_ID_EMBEDDED_ERROR = M_ID_OFFSET_EMBEDDED,
};

struct Embedded_Status {
    uint32_t uptime;

    float mem_capacity;

    /* TODO: add more embedded status measurements. */
};

extern const Transaction tEmbeddedStatus;
extern const Error eEmbeddedError;

#endif
