/*
 * Testing program for IO library functionality.
 */

#include <qubobus.h>
#include <io.h>

#if QUBOBUS_PROTOCOL_VERSION != 2
#error Update me with new message defs!
#endif

#include <stdio.h>

int main() { 
    int success = 1;

    //TODO: test the io functionality for proper operation. 

    if (success) {
        printf("No unrecoverable errors found!\n");
    }
    return !success;
}
