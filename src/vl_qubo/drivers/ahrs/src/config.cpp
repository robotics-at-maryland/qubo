
#include "../include/util.h"
#include <string>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
   if (argc == 3) {
      try {
         AHRS *ahrs = new AHRS(std::string(argv[1]),getBaudrate(argv[2]));
         AHRS::AcqConfig config;
         AHRS::FilterData filters;
         long unsigned int i;

         ahrs->openDevice();
         printf("Connected to %s.\n", (ahrs->getInfo()).c_str());

         ahrs->sendAHRSDataFormat();

         printf("AHRS Configuration data:\n");

         config = ahrs->getAcqConfig();
         printf("Acqusition mode:   %d\n", config.poll_mode);
         printf("Flush Filter:      %d\n", config.flush_filter);
         printf("Sample delay:      %f\n", config.sample_delay);

         filters = ahrs->getFIRFilters();
         printf("Number of filters: %lu\n", filters.size());
         for (i=0; i < filters.size(); i++)
         printf("Filter #%.2lu:        %f\n",i, filters[i]);

         printf("Mag Truth Method:  %d\n", ahrs->getMagTruthMethod());
         printf("Functional mode:   %d\n", ahrs->getAHRSMode());
         printf("Declination:       %f\n", ahrs->getDeclination());
         printf("UserCalNumPoints:  %d\n", ahrs->getCalPoints());
         printf("Mag Setting:       %d\n", ahrs->getMagCalID());
         printf("Accel setting:     %d\n", ahrs->getAccelCalID());
         printf("Mounting mode:     %d\n", ahrs->getMounting());
         printf("Baudrate:          %d\n", ahrs->getBaudrate().baud);
         printf("True north?        %d\n", ahrs->getTrueNorth());
         printf("Big endian?        %d\n", ahrs->getBigEndian());
         printf("Auto sampling?     %d\n", ahrs->getAutoCalibration());
         printf("Mils/Degrees?      %d\n", ahrs->getMils());
         printf("HPR During Cal?    %d\n", ahrs->getHPRCal());
         ahrs->closeDevice();
         exit(0);
      }
      catch (std::exception& e)
      {
         printError(e);
      }
   } 
   printUsage();
}
