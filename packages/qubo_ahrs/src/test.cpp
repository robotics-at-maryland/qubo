
#include "../include/IMU.h"
#include <string>
#include <stdio.h>
#include <unistd.h>

int main()
{
   IMU *imu = new IMU("/dev/ttyUSB0");
   IMUData data;
   printf("Opening Device\n");
   imu->openDevice();
   printf("Device %s\n", (imu->getInfo()).c_str());
   imu->sendIMUDataFormat();
   while (imu->isOpen()) {
      // Poll data
      data = imu->pollIMUData();
      // Print the data formatted.
      printf("Q: (%5.0f,%5.0f,%5.0f,%5.0f)\n",
            data.quaternion[0], data.quaternion[1], data.quaternion[2], data.quaternion[3]);
      printf("G: (%.3f,%.3f,%.3f)\n",
            data.gyroX, data.gyroY, data.gyroZ);
      printf("A: (%.3f,%.3f,%.3f)\n",
            data.accelX, data.accelY, data.accelZ);
      printf("M: (%.3f,%.3f,%.3f)\n",
            data.magX, data.magY, data.magZ);
      // Sleep to fill out 30hz.
      usleep(100000);
   }
   imu->closeDevice();
}
