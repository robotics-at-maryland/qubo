#include <depth.h>

const Transaction tDepthStatus = {
    .name = "Depth Status",
    .id = M_ID_DEPTH_STATUS,
    .request = EMPTY,
    .response = sizeof(struct Depth_Status),
};

const Transaction tDepthMonitorEnable = {
    .name = "Depth Monitor Enable",
    .id = M_ID_DEPTH_MONITOR_ENABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tDepthMonitorDisable = {
    .name = "Depth Monitor Disable",
    .id = M_ID_DEPTH_MONITOR_DISABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tDepthMonitorSetConfig = {
    .name = "Depth Monitor Set Config",
    .id = M_ID_DEPTH_MONITOR_SET_CONFIG,
    .request = sizeof(struct Depth_Monitor_Config),
    .response = EMPTY,
};

const Transaction tDepthMonitorGetConfig = {
    .name = "Depth Monitor Get Config",
    .id = M_ID_DEPTH_MONITOR_GET_CONFIG,
    .request = sizeof(struct Depth_Monitor_Config_Request),
    .response = sizeof(struct Depth_Monitor_Config),
};

const Error eDepthUnreachable = {
    .name = "Depth Unreachable",
    .id = E_ID_DEPTH_UNREACHABLE,
    .size = EMPTY,
};
