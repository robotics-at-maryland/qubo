
#include "../include/util.h"

#include <stdio.h>
#include <exception>
#include <sysexits.h>
#include <stdlib.h>

void printUsage()
{
   printf("Usage: <imu-tty> <imu-baud>\n");
   exit(EX_USAGE);
}

DVL::DVLSpeed getBaudrate(const char* arg)
{
   throw DVLException("Unimplemented.");
   switch (atoi(arg))
   {
   }
}

void printError(std::exception& e)
{
   printf("Error occured: %s\n", e.what());
}
