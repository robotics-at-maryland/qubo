#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_POWER_H
#define QUBOBUS_POWER_H

enum {
    M_ID_POWER_STATUS = M_ID_OFFSET_POWER,

    M_ID_POWER_RAIL_ENABLE,

    M_ID_POWER_RAIL_DISABLE,

    M_ID_POWER_MONITOR_ENABLE,

    M_ID_POWER_MONITOR_DISABLE,

    M_ID_POWER_MONITOR_SET_CONFIG,

    M_ID_POWER_MONITOR_GET_CONFIG

};

enum {
    E_ID_POWER_UNREACHABLE = M_ID_OFFSET_POWER,
};

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

struct Power_Rail {
    uint8_t rail_id;
};

struct Power_Status {
    float voltage;
    float current;

    uint8_t rail_id;
    uint8_t warning_level;
};

struct Power_Monitor_Config_Request {
    uint8_t rail_id;
    uint8_t warning_level;
};

struct Power_Monitor_Config {
    float voltage[2];
    float current[2];
    
    uint8_t rail_id;
    uint8_t warning_level;
};

extern const Transaction tPowerStatus;
extern const Transaction tPowerRailEnable;
extern const Transaction tPowerRailDisable;
extern const Transaction tPowerMonitorEnable;
extern const Transaction tPowerMonitorDisable;
extern const Transaction tPowerMonitorSetConfig;
extern const Transaction tPowerMonitorGetConfig;
extern const Error ePowerUnreachable;

#endif
