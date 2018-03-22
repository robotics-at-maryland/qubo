#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_THRUSTER_H
#define QUBOBUS_THRUSTER_H

enum {
    M_ID_THRUSTER_SET = M_ID_OFFSET_THRUSTER,

    M_ID_THRUSTER_STATUS,

    M_ID_THRUSTER_SET_CONFIG,

    M_ID_THRUSTER_GET_CONFIG,

    M_ID_THRUSTER_MONITOR_ENABLE,

    M_ID_THRUSTER_MONITOR_DISABLE,

    M_ID_THRUSTER_MONITOR_SET_CONFIG,

    M_ID_THRUSTER_MONITOR_GET_CONFIG,
};

enum {
    E_ID_THRUSTER_UNREACHABLE = M_ID_OFFSET_THRUSTER,
};

struct Thruster_Set {
    float throttle;
    uint8_t thruster_id;
};

struct Thruster_Status_Request {
    uint8_t thruster_id;
};

struct Thruster_Status {
    float thruster_A;
    float thruster_V;
    float throttle_setting;
    int16_t pwm_duty_cycle;
};

struct Thruster_Config {
    float thruster_gain;
};

struct Thruster_Monitor_Config {
    float thruster_high_A;
    float thruster_low_V;

    uint8_t warning_level;
};

extern const Transaction tThrusterSet;
extern const Transaction tThrusterStatus;
extern const Transaction tThrusterSetConfig;
extern const Transaction tThrusterGetConfig;
extern const Transaction tThrusterMonitorEnable;
extern const Transaction tThrusterMonitorDisable;
extern const Transaction tThrusterMonitorSetConfig;
extern const Transaction tThrusterMonitorGetConfig;
extern const Error eThrusterUnreachable;

#endif
