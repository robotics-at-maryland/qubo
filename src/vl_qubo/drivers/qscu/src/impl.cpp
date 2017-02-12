// UNIX Serial includes
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdarg.h>

// Header include
#include "QSCU.h"

#if QUBOBUS_PROTOCOL_VERSION != 3
#error Update me with new message defs!
#endif

#include "impl.cpp"

#include <stdio.h>

