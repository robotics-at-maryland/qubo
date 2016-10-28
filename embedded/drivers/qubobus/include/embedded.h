#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_EMBEDDED_H
#define QUBOBUS_EMBEDDED_H

enum {
    /* Message requesting a status message from the embedded system. */
    D_ID_EMBEDDED_STATUS_REQUEST = M_ID_OFFSET_EMBEDDED,

    /* Message containing info about the current system state. */
    D_ID_EMBEDDED_STATUS
};

enum {
    /* Error sent when the embedded system cannot allocate memory. */
    E_ID_EMBEDDED_MEMORY = M_ID_OFFSET_EMBEDDED,

    /* Error sent when a page fault or general protection fault occurrs. */
    E_ID_EMBEDDED_PAGE_FAULT,

    /* Error sent when an assert fails in an embedded task. */
    E_ID_EMBEDDED_ASSERT
};

struct Embedded_Status {
    /* Current uptime of the embedded system, in seconds. */
    uint32_t uptime;

    /* Fraction of memory currently allocated. */
    float mem_capacity;

    /* TODO: add more embedded status measurements. */
};

#endif
