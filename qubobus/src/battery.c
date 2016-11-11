#include <battery.h>

const Transaction tBatteryStatus = {
    .name = "Battery Status",
    .id = M_ID_BATTERY_STATUS,
    .request = sizeof(struct Battery_ID),
    .response = sizeof(struct Battery_Status)
};

const Transaction tBatteryShutdown = {
    .name = "Battery Shutdown",
    .id = M_ID_BATTERY_SHUTDOWN,
    .request = sizeof(struct Battery_ID),
    .response = EMPTY,
};

const Transaction tBatteryMonitorEnable = {
    .name = "Battery Monitor Enable",
    .id = M_ID_BATTERY_MONITOR_ENABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tBatteryMonitorDisable = {
    .name = "Battery Monitor Disable",
    .id = M_ID_BATTERY_MONITOR_DISABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tBatteryMonitorSetConfig = {
    .name = "Battery Monitor Set Config",
    .id = M_ID_BATTERY_MONITOR_SET_CONFIG,
    .request = sizeof(struct Battery_Monitor_Config),
    .response = EMPTY,
};

const Transaction tBatteryMonitorGetConfig = {
    .name = "Battery Monitor Get Config",
    .id = M_ID_BATTERY_MONITOR_GET_CONFIG,
    .request = EMPTY,
    .response = sizeof(struct Battery_Monitor_Config),
};

const Error eBatteryUnreachable = {
    .name = "Battery Unreachable",
    .id = E_ID_BATTERY_UNREACHABLE,
    .size = EMPTY,
};
