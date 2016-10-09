#include <stdint.h>

#ifndef QUBOBUS_SAFETY_H
#define QUBOBUS_SAFETY_H

#define D_ID_SAFETY_STATUS_REQUEST 20

#define D_ID_SAFETY_STATUS 21
struct Safety_Status {
    uint8_t hardware_sw;
    uint8_t sofware_sw;
};

/* Host -> QSCU Command to make the robot safe to handle. */
#define D_ID_SAFETY_SET_SAFE 22

/* Host -> QSCU Command to make the robot unsafe */
#define D_ID_SAFETY_SET_UNSAFE 23

#endif
