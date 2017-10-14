#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_BATTERY_H
#define QUBOBUS_BATTERY_H

enum {
    M_ID_BATTERY_STATUS = M_ID_OFFSET_BATTERY,

    M_ID_BATTERY_SHUTDOWN,
        
    M_ID_BATTERY_MONITOR_ENABLE,

    M_ID_BATTERY_MONITOR_DISABLE,

    M_ID_BATTERY_MONITOR_SET_CONFIG,

    M_ID_BATTERY_MONITOR_GET_CONFIG
};

enum {
    E_ID_BATTERY_UNREACHABLE = M_ID_OFFSET_BATTERY,
};

enum {
    BATTERY_0,
    BATTERY_1,
};

struct Battery_ID {
    uint8_t battery_id;
};

struct Battery_Status {
    float voltage;
    float humidity;
    float pressure;
    float temperature;
    float hydrogen;

    uint8_t battery_id;
};


struct Battery_Monitor_Config {
    float voltage[2];
    float humidity[2];
    float pressure[2];
    float temperature[2];
    float hydrogen[2];

    uint8_t warning_level;
};

extern const Transaction tBatteryStatus;
extern const Transaction tBatteryShutdown;
extern const Transaction tBatteryMonitorEnable;
extern const Transaction tBatteryMonitorDisable;
extern const Transaction tBatteryMonitorSetConfig;
extern const Transaction tBatteryMonitorGetConfig;
extern const Error eBatteryUnreachable;

#endif
