
#include "../include/util.h"
#include <string>
#include <iostream>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3)
   {
      DVL *dvl = NULL;
      DVL::DVLData data;
      clock_t curr, last;
      double hz;
      try {
         dvl = new DVL(std::string(argv[1]),getBaudrate(argv[2]));
         dvl->openDevice();
         std::cout << dvl->getSystemInfo() << std::endl;
         dvl->enableMeasurement();
         last = clock();
         while (dvl->isOpen()) {
               curr = clock();
               data = dvl->getDVLData();
               std::cout 
                   << data.mms_east << "/" 
                   << data.mms_north << "/" 
                   << data.mms_surface << std::endl;
               hz = CLOCKS_PER_SEC / ((double)(curr-last))/10;
               std::cout << "Polling at " << hz << " Hz" << std::endl;
               last = curr;
         }
      } catch (std::exception& e) {
          dvl->closeDevice();
          delete dvl;
         printError(e);
         return DVL_ERR;
      }
      if (dvl != NULL) {
          dvl->closeDevice();
          delete dvl;
      }
      return 0;
   }
   printUsage();
}
