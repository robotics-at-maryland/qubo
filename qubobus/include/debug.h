#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_DEBUG_H
#define QUBOBUS_DEBUG_H

enum {
    M_ID_DEBUG_LOG_READ = M_ID_OFFSET_DEBUG,

    M_ID_DEBUG_LOG_ENABLE,

    M_ID_DEBUG_LOG_DISABLE,
    
    /* TODO: add more debugging operations. */
};

enum {
    E_ID_DEBUG_LOG_ERROR = M_ID_OFFSET_DEBUG,

    /* TODO: determine what kinds of errors can occur during debugging operations. */
};

struct Log_Read_Request {
    uint32_t block_id;
};

struct Log_Block {
    char data[512];
};

extern const Transaction tDebugLogRead;
extern const Transaction tDebugLogEnable;
extern const Transaction tDebugLogDisable;
extern const Error eDebugLogError;

#endif
