
#include "../include/util.h"

#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3) {
      try {
         IMU *imu = new IMU(std::string(argv[1]),getBaudrate(argv[2]));
         char *buffer = NULL;
         size_t bufsize;
         bool success;
         // Connect to the IMU.
         imu->openDevice();
         printf("Connected to %s.\n", (imu->getInfo()).c_str());
         // Prompt the user for a new baudrate.
         printf("New speed? (baud) \n");
         if (getline(&buffer, &bufsize, stdin) == -1)
         {
            fprintf(stderr, "Unable to read in new baudrate.\n");
            imu->closeDevice();
            return -1;
         }
         // Attempt to read in a baudrate and change to it.
         success = imu->changeSpeed(getBaudrate(buffer));
         // Clean up and close resources.
         free(buffer);
         imu->closeDevice();
         // Tell the user of the result.
         if (success) {
            fprintf(stderr, "Changed speed successfully.\n");
            return 0;
         } else {
            fprintf(stderr, "Unable to change IMU speed.\n");
            return 1;
         }

      } catch (std::exception& e) {
         printError(e);
      }
   }
   printUsage();
}
