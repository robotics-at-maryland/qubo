
#include "../include/util.h"
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3)
   {
      DVL *dvl;
      DVL::DVLData data;
      clock_t curr, last;
      double hz;
      try {
         dvl = new DVL(std::string(argv[1]),getBaudrate(argv[2]));
         std::cout << dvl->getSystemInfo() << std::endl;
         last = clock();
         while (dvl->isOpen()) {
               curr = clock();
               data = dvl->getDVLData();
               hz = CLOCKS_PER_SEC / ((double)(curr-last))/10;
               std::cout << "Polling at " << hz << " Hz" << std::endl;
               last = curr;
         }
      } catch (std::exception& e) {
         delete dvl;
         printError(e);
         return DVL_ERR;
      }
   }
   printUsage();
}
