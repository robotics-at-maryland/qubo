#include <thruster.h>

const Transaction tThrusterSet = {
    .name = "Thruster Set",
    .id = M_ID_THRUSTER_SET,
    .request = sizeof(struct Thruster_Set),
    .response = EMPTY,
};

const Transaction tThrusterStatus = {
    .name = "Thruster Status",
    .id = M_ID_THRUSTER_STATUS,
    .request = sizeof(struct Thruster_Status_Request),
    .response = sizeof(struct Thruster_Status),
};

const Transaction tThrusterSetConfig = {
    .name = "Thruster Set Config",
    .id = M_ID_THRUSTER_SET_CONFIG,
    .request = sizeof(struct Thruster_Config),
    .response = EMPTY,
};

const Transaction tThrusterGetConfig = {
    .name = "Thruster Get Config",
    .id = M_ID_THRUSTER_GET_CONFIG,
    .request = EMPTY,
    .response = sizeof(struct Thruster_Config),
};

const Transaction tThrusterMonitorEnable = {
    .name = "Thruster Monitor Enable",
    .id = M_ID_THRUSTER_MONITOR_ENABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tThrusterMonitorDisable = {
    .name = "Thruster Monitor Disable",
    .id = M_ID_THRUSTER_MONITOR_DISABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tThrusterMonitorSetConfig = {
    .name = "Thruster Monitor Set Config",
    .id = M_ID_THRUSTER_MONITOR_SET_CONFIG,
    .request = sizeof(struct Thruster_Monitor_Config),
    .response = EMPTY,
};

const Transaction tThrusterMonitorGetConfig = {
    .name = "Thruster Monitor Get Config",
    .id = M_ID_THRUSTER_MONITOR_GET_CONFIG,
    .request = EMPTY,
    .response = sizeof(struct Thruster_Monitor_Config),
};

const Error eThrusterUnreachable = {
    .name = "Thruster Unreachable",
    .id = E_ID_THRUSTER_UNREACHABLE,
    .size = EMPTY,
};
