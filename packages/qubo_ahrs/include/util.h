#ifndef IMU_UTIL_H
#define IMU_UTIL_H

#include "IMU.h"

/** UNIX error code to return to the OS. */
#define IMU_ERR 44

/** Prints the script usage arguments. */
void printUsage();

/** Converts a string into an IMUSpeed */
IMUSpeed getBaudrate(const char*);

/** Prints an exception and kills the program. */
void printError(std::exception& e);

#endif
