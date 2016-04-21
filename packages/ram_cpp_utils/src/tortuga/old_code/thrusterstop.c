/*
 * Copyright (C) 2007 Robotics at Maryland
 * Copyright (C) 2007 Steve Moskovchenko <stevenm@umd.edu>
 * All rights reserved.
 *
 * Author: Steve Moskovchenko <stevenm@umd.edu>
 * File:  packages/thrusterapi/src/thrusterstop.c
 */

// STD Includes
#include <stdio.h>

// Unix Includes
#include <unistd.h>

// Project Includes
#include "tortuga/thrusterapi.h"

int main(int argc, char ** argv)
{
    int fd = openThrusters("/dev/motor");

//    int fd = open("/dev/ttyUSB0", O_RDWR);
//    printf("\nSetting address: %d\n", writeReg(fd, 1, REG_ADDR, 4));

    int i=0;

//    int err=0;
//    int e1=0, e2=0, e3=0, e4=0;
    setReg(fd, 1, REG_TIMER, 0);
    setReg(fd, 2, REG_TIMER, 0);
    setReg(fd, 3, REG_TIMER, 0);
    setReg(fd, 4, REG_TIMER, 0);


    for(i=0; i<10; i++)
    {
        printf("\n");
        printf("\nResult 1 is: %d\n", setSpeed(fd, 1, 0));
        printf("\nResult 2 is: %d\n", setSpeed(fd, 2, 0));
        printf("\nResult 3 is: %d\n", setSpeed(fd, 3, 0));
        printf("\nResult 4 is: %d\n", setSpeed(fd, 4, 0));
    }

    fsync(fd);
    printf("\n");
    return 0;
}

