#include <stdint.h>

#ifndef QUBOBUS_EMBEDDED_H
#define QUBOBUS_EMBEDDED_H

/* Host -> QSCU Message requesting a status message from the embedded system. */
#define D_ID_EMBEDDED_STATUS_REQUEST 10

/* QSCU -> Host Message containing info about the current system state. */
#define D_ID_EMBEDDED_STATUS 11
struct Embedded_Status {
    /* Current uptime of the embedded system, in seconds. */
    uint32_t uptime;

    /* Current CPU load of the embedded system. */
    float cpu_load;

    /* Fraction of memory currently allocated. */
    float mem_capacity;

    /* TODO: add more embedded status measurements. */
};

/* Error sent when the embedded system passes a maximum load threshold. */
#define E_ID_EMBEDDED_OVERLOAD 10

/* Error sent when the embedded system cannot allocate memory. */
#define E_ID_EMBEDDED_MEMORY 11

/* Error sent when a page fault or general protection fault occurrs. */
#define E_ID_EMBEDDED_PAGE_FAULT 12

/* Error sent when an assert fails in an embedded task. */
#define E_ID_EMBEDDED_ASSERT 13

#endif
