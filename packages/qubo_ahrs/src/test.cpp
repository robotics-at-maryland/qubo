
#include "../include/IMU.h"
#include <string>
#include <stdio.h>

int main()
{
   IMU *imu = new IMU("/dev/ttyUSB0");
   int err = 0;
   char info[9];
   printf("Opening Device\n");
   err = imu->openDevice();
   printf("Err status: %d\n", err);
   err = imu->getInfo(info);
   printf("Err status: %d\n", err);
   info[8] = '\0';
   printf("Got response (%s)\n", info);
   imu->closeDevice();
}
