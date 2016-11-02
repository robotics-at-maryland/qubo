
#ifndef QUBOBUS_MODULES_H
#define QUBOBUS_MODULES_H

/* Mesage id offsets for different modules */
enum {
    M_ID_OFFSET_MIN = 0,
    M_ID_OFFSET_CORE = 1,
    M_ID_OFFSET_EMBEDDED = 10,
    M_ID_OFFSET_SAFETY = 20,
    M_ID_OFFSET_BATTERY = 30,
    M_ID_OFFSET_POWER = 40,
    M_ID_OFFSET_THRUSTER = 50,
    M_ID_OFFSET_PNEUMATICS = 60,
    M_ID_OFFSET_DEPTH = 70,
    M_ID_OFFSET_MAX = 80,
};

#define IS_MESSAGE_ID(X) ((M_ID_OFFSET_MIN < (X)) && ((X) < M_ID_OFFSET_MAX))

#endif
