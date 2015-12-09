
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
      try {
         //dvl = new DVL(std::string(argv[1]),getBaudrate(argv[2]));
         clock_t curr, last;
         double hz;
         last = clock();
         while (dvl->isOpen()) {
            try {
               curr = clock();
               hz = CLOCKS_PER_SEC / ((double)(curr-last))/10;
               printf("Polling at %.2f Hz\n", hz);
               last = curr;
            } catch (DVLException& e) {
            }
         }
      } catch (std::exception& e) {
         printError(e);
         return DVL_ERR;
      }
   }
   printUsage();
}
