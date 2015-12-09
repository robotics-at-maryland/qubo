
#include "../include/util.h"
#include <string>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
   if (argc == 3) {
      try {
         //DVL *dvl = new DVL(std::string(argv[1]),getBaudrate(argv[2]));
         exit(0);
      }
      catch (std::exception& e)
      {
         printError(e);
      }
   } 
   printUsage();
}
