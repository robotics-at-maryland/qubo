
#include "../include/util.h"
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3)
   {
      IMU *imu;
      try {
         imu = new IMU(std::string(argv[1]),getBaudrate(argv[2]));
         IMUData data;
         clock_t curr, last;
         double hz;
         imu->openDevice();
         printf("Connected to %s.\n", (imu->getInfo()).c_str());
         imu->sendIMUDataFormat();
         last = clock();
         while (imu->isOpen()) {
            try {
               // Poll data
               data = imu->pollIMUData();
               // Print the data formatted.
               printf("Q: (%1.3f,%1.3f,%1.3f,%1.3f)\n",
                     data.quaternion[0], data.quaternion[1], data.quaternion[2], data.quaternion[3]);
               printf("G: (%.3f,%.3f,%.3f)\n",
                     data.gyroX, data.gyroY, data.gyroZ);
               printf("A: (%.3f,%.3f,%.3f)\n",
                     data.accelX, data.accelY, data.accelZ);
               printf("M: (%.3f,%.3f,%.3f)\n",
                     data.magX, data.magY, data.magZ);
               curr = clock();
               hz = CLOCKS_PER_SEC / ((double)(curr-last))/10;
               printf("Polling at %.2f Hz\n", hz);
               last = curr;
            } catch (IMUException& e) {
               imu->closeDevice();
               imu->openDevice();
            }
         }
         imu->closeDevice();
      } catch (std::exception& e) {
         delete imu;
         printError(e);
         return IMU_ERR;
      }
   }
   printUsage();
}
