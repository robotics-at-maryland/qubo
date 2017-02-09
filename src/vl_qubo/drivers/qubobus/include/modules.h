#include <unistd.h>

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
    M_ID_OFFSET_DEBUG = 80,
    M_ID_OFFSET_MAX = 90,
};

#define IS_MESSAGE_ID(X) ((M_ID_OFFSET_MIN < (X)) && ((X) < M_ID_OFFSET_MAX))

/* Different monitor severity levels for managing hardware parameters. */
enum {
    MONITOR_LEVEL_NORMAL = 0,
    MONITOR_LEVEL_WARNING,
    MONITOR_LEVEL_CRITICAL,
};

/* Indexes in the limit arrays for monitor configurations. */
enum {
    LLIMIT = 0,
    HLIMIT = 1,
};

#define EMPTY 0

/* Definition of associated messages that are expected in a request/response format. */
typedef struct _Transaction {
    char const *name;
    size_t request;
    size_t response;
    uint8_t id;
} Transaction;

/* Definition of an error message. */
typedef struct _Error {
    char const *name;
    size_t size;
    uint8_t id;
} Error;


#endif
