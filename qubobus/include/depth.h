#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_DEPTH_H
#define QUBOBUS_DEPTH_H

enum {
    M_ID_DEPTH_STATUS = M_ID_OFFSET_DEPTH,

    M_ID_DEPTH_MONITOR_ENABLE,

    M_ID_DEPTH_MONITOR_DISABLE,

    M_ID_DEPTH_MONITOR_SET_CONFIG,
        
    M_ID_DEPTH_MONITOR_GET_CONFIG
};

enum {
    E_ID_DEPTH_UNREACHABLE = M_ID_OFFSET_DEPTH,
};

struct Depth_Status {
    float depth_m;

    uint8_t warning_level;
};

struct Depth_Monitor_Config_Request {
    uint8_t warning_level;
};

struct Depth_Monitor_Config {
    float depth[2];
    
    uint8_t warning_level;
};

extern const Transaction tDepthStatus;
extern const Transaction tDepthMonitorEnable;
extern const Transaction tDepthMonitorDisable;
extern const Transaction tDepthMonitorSetConfig;
extern const Transaction tDepthMonitorGetConfig;
extern const Error eDepthUnreachable;

#endif
