#ifndef DVL_UTIL_H
#define DVL_UTIL_H

#include "DVL.h"

/** UNIX error code to return to the OS. */
#define DVL_ERR 45

/** Prints the script usage arguments. */
void printUsage();

/** Converts a string into an IMUSpeed */
DVL::DVLSpeed getBaudrate(const char*);

/** Prints an exception and kills the program. */
void printError(std::exception& e);

#endif
