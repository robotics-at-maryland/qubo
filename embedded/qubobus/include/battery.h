#include <stdint.h>

#ifndef QUBOBUS_BATTERY_H
#define QUBOBUS_BATTERY_H

/* Host -> QSCU requesting back a Battery_Status message */
#define D_ID_BATTERY_STATUS_REQUEST 30
struct Battery_Status_Request {
    uint8_t batt_id;
};

/* QSCU -> Host replying to a Battery_Status_Request message */
#define D_ID_BATTERY_STATUS 31
struct Battery_Status {
    uint8_t batt_id;

    /* TODO: add sensor data fields. */
};

/* Host -> QSCU requesting that a battery be shut off immediately. */
#define D_ID_BATTERY_SHUTDOWN 32
struct Battery_Shutdown {
    uint8_t batt_id;
};

/* Error sent back when the battery hardware is unreachable*/
#define ERR_ID_BATTERY_UNREACHABLE 30

#endif
