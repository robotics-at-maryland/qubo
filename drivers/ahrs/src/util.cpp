
#include "../include/util.h"

#include <stdio.h>
#include <exception>
#include <sysexits.h>
#include <stdlib.h>

void printUsage()
{
   printf("Usage: <ahrs-tty> <ahrs-baud>\n");
   exit(EX_USAGE);
}

AHRS::AHRSSpeed getBaudrate(const char* arg)
{
   switch (atoi(arg))
   {
      case 2400:
         return AHRS::k2400;
      case 4800:
         return AHRS::k4800;
      case 9600:
         return AHRS::k9600;
      case 19200:
         return AHRS::k19200;
      case 38400:
         return AHRS::k38400;
      case 57600:
         return AHRS::k57600;
      case 115200:
         return AHRS::k115200;
      default:
         return AHRS::k0;
   }
}

void printError(std::exception& e)
{
   printf("Error occured: %s\n", e.what());
}
