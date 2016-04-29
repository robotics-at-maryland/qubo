
#include "../include/util.h"

#include <stdio.h>
#include <exception>
#include <sysexits.h>
#include <stdlib.h>

void printUsage()
{
   printf("Usage: <dvl-tty> <dvl-baud>\n");
   exit(EX_USAGE);
}

DVL::DVLSpeed getBaudrate(const char* arg)
{
   switch (atoi(arg))
   {
      case 300:
         return DVL::k300;
      case 1200:
         return DVL::k1200;
      case 2400:
         return DVL::k2400;
      case 4800:
         return DVL::k4800;
      case 9600:
         return DVL::k9600;
      case 19200:
         return DVL::k19200;
      case 38400:
         return DVL::k38400;
      case 57600:
         return DVL::k57600;
      case 115200:
         return DVL::k115200;
      default:
         throw DVLException("Unknown baudrate specified");
   }
}

void printError(std::exception& e)
{
   printf("Error occured: %s\n", e.what());
}
