#include <stdint.h>

#ifndef QUBOBUS_EMBEDDED_H
#define QUBOBUS_EMBEDDED_H

/* Host -> QSCU Request for a status reading. */
#define D_ID_EMBEDDED_STATUS_REQUEST 10

/* QSCU -> Host A status reading containing info about the current system state. */
#define D_ID_EMBEDDED_STATUS 11
struct Embedded_Status {
    /* Current uptime of the embedded system, in seconds. */
    uint32_t uptime;
    /* Current CPU load of the embedded system. */
    float cpu_load;
    /* Current number of active tasks running. */
    uint16_t num_tasks;
    /* TODO: add more embedded status measurements. */
};

#endif
