
#include "../include/util.h"
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3)
   {
      AHRS *ahrs;
      try {
         ahrs = new AHRS(std::string(argv[1]),getBaudrate(argv[2]));
         AHRS::AHRSData data;
         clock_t curr, last;
         double hz;
         ahrs->openDevice();
         printf("Connected to %s.\n", (ahrs->getInfo()).c_str());
         ahrs->sendAHRSDataFormat();
         last = clock();
         while (ahrs->isOpen()) {
            try {
               // Poll data
               data = ahrs->pollAHRSData();
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
            } catch (AHRSException& e) {
               ahrs->closeDevice();
               ahrs->openDevice();
            }
         }
         ahrs->closeDevice();
      } catch (std::exception& e) {
         delete ahrs;
         printError(e);
         return AHRS_ERR;
      }
   }
   printUsage();
}
