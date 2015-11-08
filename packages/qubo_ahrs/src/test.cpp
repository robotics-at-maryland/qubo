
#include "../include/util.h"
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3)
   {
      try {
         IMU *imu = new IMU(std::string(argv[1]),getBaudrate(argv[2]));
         IMUData data;
         clock_t curr, last;
         double hz;
         imu->openDevice();
         printf("Connected to %s.\n", (imu->getInfo()).c_str());
         imu->sendIMUDataFormat();
         last = clock();
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
            curr = clock();
            hz = CLOCKS_PER_SEC / ((double)(curr-last))/10;
            printf("Poll took %.6f seconds (%.2f Hz)\n", 1/hz, hz);
            last = curr;
         }
         imu->closeDevice();
      } catch (std::exception& e) {
         printError(e);
      }
   }
   printUsage();
}
