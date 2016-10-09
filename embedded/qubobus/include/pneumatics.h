#include <stdint.h>

#ifndef QUBOBUS_PNEUMATICS_H
#define QUBOBUS_PNEUMATICS_H

/* Host -> QSCU message containing a mode setting for a single valve. */
#define D_ID_PNEUMATICS_SET 60
struct Pneumatics_Set {
    uint8_t valve_id;
    uint8_t mode;
};

/* Error message sent when a thruster is not connected. */
#define ERR_ID_PNEUMATICS_UNREACHABLE 60

#endif
