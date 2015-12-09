
#include "../include/util.h"

#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[])
{
   if (argc == 3) {
      try {
         //DVL *dvl = new DVL(std::string(argv[1]),getBaudrate(argv[2]));
         char *buffer = NULL;
         size_t bufsize;
         // Prompt the user for a new baudrate.
         printf("New speed? (baud) \n");
         if (getline(&buffer, &bufsize, stdin) == -1)
         {
            fprintf(stderr, "Unable to read in new baudrate.\n");
            return -1;
         }
         free(buffer);
         // Tell the user of the result.
         fprintf(stderr, "Changed speed successfully.\n");
         exit(0);
      } catch (std::exception& e) {
         printError(e);
      }
   }
   printUsage();
}
