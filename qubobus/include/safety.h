#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_SAFETY_H
#define QUBOBUS_SAFETY_H

enum {
    M_ID_SAFETY_STATUS = M_ID_OFFSET_SAFETY,

    M_ID_SAFETY_SET_SAFE,

    M_ID_SAFETY_SET_UNSAFE
};

struct Safety_Status {
    uint8_t hardware_sw;
    uint8_t sofware_sw;
};

extern const Transaction tSafetyStatus;
extern const Transaction tSafetySetSafe;
extern const Transaction tSafetySetUnsafe;

#endif
