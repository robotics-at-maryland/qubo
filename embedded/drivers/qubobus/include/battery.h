#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_BATTERY_H
#define QUBOBUS_BATTERY_H

enum {
    /* Command requsting a battery status report. */
    D_ID_BATTERY_STATUS_REQUEST = M_ID_OFFSET_BATTERY,

    /* Response with status information about the battery. */
    D_ID_BATTERY_STATUS,

    /* Command to shut the battery off immediately. */
    D_ID_BATTERY_SHUTDOWN,
        
    /* Command to enable background monitoring */
    D_ID_BATTERY_MONITOR_ENABLE,

    /* Command to disable background monitoring. */
    D_ID_BATTERY_MONITOR_DISABLE,

    /* Message to configure background monitoring */
    D_ID_BATTERY_MONITOR_CONFIG
};

enum {
    /* Error sent when the battery hardware is unreachable. */
    E_ID_BATTERY_UNREACHABLE = M_ID_OFFSET_BATTERY,

    /* Error sent when the battery is disconnected. */
    E_ID_BATTERY_DISCONNECT,

    /* Error sent by the background monitor for soft limits being exceeded. */
    E_ID_BATTERY_MONITOR_SOFT_WARNING,

    /* Error sent from background monitoring for hard limits being exceeded. */
    E_ID_BATTERY_MONITOR_HARD_WARNING
};

enum {
    BATTERY_0,
    BATTERY_1,
};

/* Payload set as a request for a battery status message. */
struct Battery_Status_Request {
    uint8_t batt_id;
};

/* Payload of a status message with information from sensors from the battery support board. */
struct Battery_Status {
    /* TODO: finalize data types for sensor data fields. */
    float voltage;
    float pressure;
    float water;
    float hydrogen;
    float temperature;

    uint8_t batt_id;
};

/* Configuration details for the background monitoring system */
struct Battery_Monitor_Config {
    /* Soft limits that generate warnings when exceeded */
    float soft_low_voltage;
    float soft_high_voltage;
    float soft_pressure_high;
    float soft_water_high;
    float soft_hydrogen_high;
    float soft_temperature_high;

    /* Hard limits that generate more important errors when exceeded */
    float hard_low_voltage;
    float hard_high_voltage;
    float hard_pressure_high;
    float hard_water_high;
    float hard_hydrogen_high;
    float hard_temperature_high;
};

/* Message detailing what battery to shut down */
struct Battery_Shutdown {
    uint8_t batt_id;
};

#endif
