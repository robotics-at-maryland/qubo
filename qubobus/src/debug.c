#include <debug.h>

const Transaction tDebugLogRead = {
    .name = "Debug Log Read",
    .id = M_ID_DEBUG_LOG_READ,
    .request = sizeof(struct Log_Read_Request),
    .response = sizeof(struct Log_Block),
};

const Transaction tDebugLogEnable = {
    .name = "Debug Log Enable",
    .id = M_ID_DEBUG_LOG_ENABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Transaction tDebugLogDisable = {
    .name = "Debug Log Disable",
    .id = M_ID_DEBUG_LOG_DISABLE,
    .request = EMPTY,
    .response = EMPTY,
};

const Error eDebugLogError = {
    .name = "Debug Log Error",
    .id = E_ID_DEBUG_LOG_ERROR,
    .size = EMPTY,
};
