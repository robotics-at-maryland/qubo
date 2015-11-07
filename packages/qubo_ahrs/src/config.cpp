
#include "../include/IMU.h"
#include <string>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
   if (argc == 2) {
      IMU *imu = new IMU(std::string(argv[1]));
      IMUConfig config;
      int i;
      try {
         printf("Connecting to device...\n");
         imu->openDevice();
         printf("Device %s found.\n", (imu->getInfo()).c_str());
         imu->sendIMUDataFormat();
         config = imu->readConfig();

         printf("IMU Configuration data:\n");
         printf("Aqusition mode:    %d\n", config.acqParams.aquisition_mode);
         printf("Flush Filter:      %d\n", config.acqParams.flush_filter);
         printf("Sample delay:      %f\n", config.acqParams.sample_delay);

         printf("Fir filter:        %d %d\n", 
               config.filters.FIRTaps.filter_id.byte_1,
               config.filters.FIRTaps.filter_id.byte_2);
         printf("# filters:         %d\n", config.filters.FIRTaps.count);
         for (i=0; i< config.filters.FIRTaps.count; i++)
         printf("Filter       #%.2d:  %f\n", i, config.filters.taps[i]);

         printf("Mag Truth Method:  %d\n", config.magTruthMethod);
         printf("Functional mode:   %d\n", config.mode);
         printf("Declination:       %f\n", config.declination.value);
         printf("UserCalNumPoints:  %d\n", config.userCalNumPoints.value);
         printf("Mag Setting:       %d\n", config.magCoeffSet.value);
         printf("Accel setting:     %d\n", config.accelCoeffSet.value);
         printf("Mounting mode:     %d\n", config.mountingRef.value);
         printf("Baudrate:          %d\n", config.baudRate.value);
         printf("True north?        %d\n", config.trueNorth.value);
         printf("Big endian?        %d\n", config.bigEndian.value);
         printf("Auto sapling?      %d\n", config.userCalAutoSampling.value);
         printf("Mils/Degrees?      %d\n", config.milOut.value);
         printf("HPR During Cal?    %d\n", config.hprDuringCal.value);
         imu->closeDevice();
      }
      catch (IMUException& e)
      {
         fprintf(stderr, "IMU Error occured: %s\n", e.what());
      }
      catch (std::exception& e)
      {
         fprintf(stderr, "Unknown error occured: %s\n", e.what());
      }
   } else {
      printf("Usage: config <imu-device-name>\n");
   }
}
