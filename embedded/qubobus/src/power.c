#include <power.h>

const Transaction tPowerStatus = {
    .name = "Power Status",
    .id = M_ID_POWER_STATUS,
    .request = sizeof(struct Power_Rail),
    .response = sizeof(struct Power_Status),
};

const Transaction tPowerRailEnable = {
    .name = "Power Rail Enable",
    .id = M_ID_POWER_RAIL_ENABLE,
    .request = sizeof(struct Power_Rail),
    .response = EMPTY,
};

const Transaction tPowerRailDisable = {
    .name = "Power Rail Disable",
    .id = M_ID_POWER_RAIL_DISABLE,
    .request = sizeof(struct Power_Rail),
    .response = EMPTY,
};

const Transaction tPowerMonitorEnable = {
    .name = "Power Monitor Enable",
    .id = M_ID_POWER_MONITOR_ENABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tPowerMonitorDisable = {
    .name = "Power Monitor Disable",
    .id = M_ID_POWER_MONITOR_DISABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tPowerMonitorSetConfig = {
    .name = "Power Monitor Set Config",
    .id = M_ID_POWER_MONITOR_SET_CONFIG,
    .request = sizeof(struct Power_Monitor_Config),
    .response = EMPTY,
};

const Transaction tPowerMonitorGetConfig = {
    .name = "Power Monitor Get Config",
    .id = M_ID_POWER_MONITOR_GET_CONFIG,
    .request = sizeof(struct Power_Monitor_Config_Request),
    .response = sizeof(struct Power_Monitor_Config),
};

const Error ePowerUnreachable = {
    .name = "Power Unreachable",
    .id = E_ID_POWER_UNREACHABLE,
    .size = EMPTY,
};
