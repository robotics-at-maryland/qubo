
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

IMUSpeed getBaudrate(const char* arg)
{
   switch (atoi(arg))
   {
      case 2400:
         return IMU::k2400;
      case 4800:
         return IMU::k4800;
      case 9600:
         return IMU::k9600;
      case 19200:
         return IMU::k19200;
      case 38400:
         return IMU::k38400;
      case 57600:
         return IMU::k57600;
      case 115200:
         return IMU::k115200;
      default:
         return IMU::k0;
   }
}

void printError(std::exception& e)
{
   printf("Error occured: %s\n", e.what());
}
