#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_PNEUMATICS_H
#define QUBOBUS_PNEUMATICS_H

enum {
    M_ID_PNEUMATICS_SET = M_ID_OFFSET_PNEUMATICS
};

enum {
    E_ID_PNEUMATICS_UNREACHABLE = M_ID_OFFSET_PNEUMATICS
};

struct Pneumatics_Set {
    uint8_t valve_id;
    uint8_t mode;
};

extern const Transaction tPneumaticsSet;
extern const Error ePneumaticsUnreachable;

#endif
