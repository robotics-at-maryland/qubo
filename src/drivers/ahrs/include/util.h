#ifndef AHRS_UTIL_H
#define AHRS_UTIL_H

#include "AHRS.h"

/** UNIX error code to return to the OS. */
#define AHRS_ERR 44

/** Prints the script usage arguments. */
void printUsage();

/** Converts a string into an AHRSSpeed */
AHRS::AHRSSpeed getBaudrate(const char*);

/** Prints an exception and kills the program. */
void printError(std::exception& e);

#endif
