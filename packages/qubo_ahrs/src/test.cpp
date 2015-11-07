
#include "../include/IMU.h"
#include <string>
#include <stdio.h>

int main()
{
   IMU *imu = new IMU("/dev/ttyUSB0");
   printf("Opening Device\n");
   imu->openDevice();
   printf("Device %s\n", (imu->getInfo()).c_str());
   imu->closeDevice();
}
