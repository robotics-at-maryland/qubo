#include <stdint.h>
#include "modules.h"

#ifndef QUBOBUS_PNEUMATICS_H
#define QUBOBUS_PNEUMATICS_H

enum {
    /* Message containing a mode setting for a single valve. */
    D_ID_PNEUMATICS_SET = M_ID_OFFSET_PNEUMATICS
};

enum {
    /* Error message sent when a thruster is not connected. */
    E_ID_PNEUMATICS_UNREACHABLE = M_ID_OFFSET_PNEUMATICS
};

/* Payload for a set message to activate or deactivate a pneumatic valve. */
struct Pneumatics_Set {
    uint8_t valve_id;
    uint8_t mode;
};

#endif
