#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_POWER_H
#define QUBOBUS_POWER_H

enum {
    /* Command requesting a status message about the power subsystem. */
    D_ID_POWER_STATUS_REQUEST = M_ID_OFFSET_POWER,

    /* Reply with information about the status of the power system. */
    D_ID_POWER_STATUS,

    /* Command to enable the power supply for a specified rail */
    D_ID_POWER_ENABLE,

    /* Command to disable the power supply for a specified rail. */
    D_ID_POWER_DISABLE,

    /* Command to enable background monitoring. */
    D_ID_POWER_MONITOR_ENABLE,

    /* Command to disable background monitoring. */
    D_ID_POWER_MONITOR_DISABLE,

    /* Command to set the software limits for determining when to send error messages. */
    D_ID_POWER_MONITOR_CONFIG

};

enum {
    /* Error sent when the power board is unreachable. */
    E_ID_POWER_UNREACHABLE = M_ID_OFFSET_POWER,

    /* Error sent when a soft limit is exceeded. */
    E_ID_POWER_MONITOR_SOFT_WARNING,

    /* Error sent when a hard limit is exceeded. */
    E_ID_POWER_MONITOR_HARD_WARNING,
};

/* Defintions for different rail IDs. */
enum {
    RAIL_ID_3V,
    RAIL_ID_5V,
    RAIL_ID_9V,
    RAIL_ID_12V,
    RAIL_ID_DVL,
    RAIL_ID_16V,
    RAIL_ID_BATT_0,
    RAIL_ID_BATT_1,
    RAIL_ID_SHORE
};

#define IS_POWER_RAIL_ID(X) ((RAIL_ID_3V <= (X)) && ((X) <= RAIL_ID_SHORE))

struct Power_Status_Request {
    uint8_t rail_id;
};

/* Payload of data about a single power rail */
struct Power_Status {
    /* Measurements for the requested common rail. */
    float rail_V;
    float rail_A;

    uint8_t rail_id;
};

struct Power_Enable {
    uint8_t rail_id;   
};

struct Power_Disable {
    uint8_t rail_id;
};

struct Power_Monitor_Config {
    /* Limits for the 3.3-volt common rail. */
    float rail_soft_low_V;
    float rail_soft_high_V;
    float rail_soft_high_A;
    
    float rail_hard_low_V;
    float rail_hard_high_V;
    float rail_hard_high_A;

    uint8_t rail_id;
};

#endif
