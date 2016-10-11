#include <stdint.h>

#ifndef QUBOBUS_BATTERY_H
#define QUBOBUS_BATTERY_H

/* Host -> QSCU Message request a read of battery status. */
#define D_ID_BATTERY_STATUS_REQUEST 30
struct Battery_Status_Request {
    uint8_t batt_id;
};

/* QSCU -> Host Message containing details about the battery. */
#define D_ID_BATTERY_STATUS 31
struct Battery_Status {
    uint8_t batt_id;

    /* TODO: finalize data types for sensor data fields. */
    float pressure;
    float humidity;
    float hydrogen;
    float temperature;
};

/* Host -> QSCU Message with configuration details for the battery system. */
#define D_ID_BATTERY_CONFIG 32
struct Battery_Config {
    float pressure_high;
    float humidity_high;
    float hydrogen_high;
    float temperature_high;
};

/* Host -> QSCU Message indicating to the battery that it should shut off immediately. */
#define D_ID_BATTERY_SHUTDOWN 33
struct Battery_Shutdown {
    uint8_t batt_id;
};

/* Error sent when the battery hardware is unreachable. */
#define E_ID_BATTERY_UNREACHABLE 30

/* Error sent when the battery is disconnected. */
#define E_ID_BATTERY_DISCONNECT 31

/* Error sent when the battery reaches an unsafe environmental state. */
#define E_ID_BATTERY_ENVIRONMENT 32

#endif
