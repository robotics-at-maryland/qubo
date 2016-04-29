/*
 * Copyright (C) 2007 Robotics at Maryland
 * Copyright (C) 2007 Steve Moskovchenko <stevenm@umd.edu>
 * All rights reserved.
 *
 * Author: Steve Moskovchenko <stevenm@umd.edu>
 * File:  packages/thrusterapi/src/thrusterapi.c
 */

// STD Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// UNIX Includes
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <poll.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

// Linux Includes (Not Sure If These Are needed)
#ifdef RAM_LINUX
  #include <bits/types.h>
  #include <linux/serial.h>
#endif // RAM_LINUX

// Project Includes
#include "tortuga/thrusterapi.h"

#define USEC_PER_MSEC 1000

/*SG: commented these out to avoid conflicting definitions from
 sensorapi.c I'm guessing this was compiled completely separately
  in the old tortuga implementation.But we compile them together,
  luckily the code will work the same because the definitions were
  identical. 
/*
int hasData(int fd, int timeout)
{
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;

    poll(&pfd, 1, timeout);

    return pfd.revents & POLLIN;
}

void miniSleep()
{
    usleep(20 * 1000);
}

int writeData(int fd, unsigned char * buf, int nbytes)
{
    int ret = write(fd, buf, nbytes);
    return ret;
}


int readData(int fd, unsigned char * buf, int nbytes)
{
    if(!hasData(fd, IO_TIMEOUT))
        return TH_IOERROR;

    return read(fd, buf, nbytes);
}
*/

int waitForData(int fd, unsigned char * buf, int nbytes, int timeout)
{
    if(!hasData(fd, timeout))
        return TH_TIMEOUTERROR;

    return read(fd, buf, nbytes);
}

void clearBuf(int fd)
{
    unsigned char c;
    while(hasData(fd, 0))
      if (1 != read(fd, &c, 1))
	break;
}


/* Returns 0 on OK */
/*
int setSpeedOld(int fd, int addr, int speed)
{
    unsigned char txData[]={0x14, 0, 0, 0, 0};
    clearBuf(fd);

    txData[1] = addr;
    txData[2] = speed & 0xFF;
    txData[3] = (speed >> 8) & 0xFF;
    txData[4] = (txData[0]+txData[1]+txData[2]+txData[3]) & 0xFF;
    writeData(fd, txData, 5);

    unsigned char resp = 0;
    if(waitForData(fd, &resp, 1, 15) != 1)
        return TH_TIMEOUTERROR;

    return resp != 0x06;
}*/


/* Returns 0 on OK */
int setSpeed(int fd, int addr, int speed)
{
    unsigned char tmp[2];
    usleep(3*1000);
    tmp[0] = speed & 0xFF;
    tmp[1] = (speed >> 8) & 0xFF;
    return multiCmd(fd, 0x14, addr, tmp, 2, 200);
}


int setVelocityLimit(int fd, int addr, int limit)
{
    unsigned char tmp[3];
    tmp[0] = 1; /* Vel Limit */
    tmp[1] = (limit & 0xFF);
    tmp[2] = (limit >> 8) & 0xFF;
    return multiCmd(fd, 0x18, addr, tmp, 3, 200);
}


int writeVelocityLimit(int fd, int addr, int limit)
{
    unsigned char tmp[3];
    tmp[0] = 1; /* Vel Limit */
    tmp[1] = (limit & 0xFF);
    tmp[2] = (limit >> 8) & 0xFF;
    return multiCmd(fd, 0x19, addr, tmp, 3, 200);
}


int setReg(int fd, int addr, int reg, int val)
{
    unsigned char tmp[3];
    tmp[0] = reg;
    tmp[1] = val;
    return multiCmd(fd, 0x18, addr, tmp, 2, 200);
}


int writeReg(int fd, int addr, int reg, int val)
{
    unsigned char tmp[3];
    tmp[0] = reg;
    tmp[1] = val;
    return multiCmd(fd, 0x19, addr, tmp, 2, 200);
}


int resetDevice(int fd, int addr)
{
    return multiCmd(fd, 0x1C, addr, NULL, 0, 200);
}



/* Internal use */
int multiCmd(int fd, int cmd, int addr, unsigned char * data, int len, int timeout)
{
    unsigned char txData[]={0, 0, 0, 0, 0, 0, 0, 0};
    clearBuf(fd);
    int i=0;

    txData[0] = cmd;
    txData[1] = addr;
    for(i=0; i<len; i++)
        txData[i+2] = data[i];

    for(i=0; i<len+2; i++)
        txData[len+2] = (txData[len+2] + txData[i]) & 0xFF;

    writeData(fd, txData, len+3);
    fsync(fd);

    unsigned char resp = 255;
    if(waitForData(fd, &resp, 1, timeout) != 1)
        return TH_TIMEOUTERROR;
    fsync(fd);
    usleep(500); /* Don't even think about it. This cannot be any lower. */
    return resp != 0x06;
}


/* Some code from cutecom, which in turn may have come from minicom */
/* FUGLY but it does what I want */
int openThrusters(const char * devName)
{
   int fd = open(devName, O_RDWR); //, O_ASYNC); // | O_ASYNC); //, O_RDWR, O_NONBLOCK);

    if(fd == -1)
        return -1;

    struct termios newtio;
    if (tcgetattr(fd, &newtio) != 0)
        printf("\nFirst stuff failed\n");

    unsigned int _baud=B19200;
    cfsetospeed(&newtio, _baud);
    cfsetispeed(&newtio, _baud);


    newtio.c_cflag = (newtio.c_cflag & ~CSIZE) | CS8;
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~(PARENB | PARODD);

    newtio.c_cflag &= ~CRTSCTS;
    newtio.c_cflag &= ~CSTOPB;

    newtio.c_iflag=IGNBRK;
    newtio.c_iflag &= ~(IXON|IXOFF|IXANY);

    newtio.c_lflag=0;

    newtio.c_oflag=0;


    newtio.c_cc[VTIME]=1;
    newtio.c_cc[VMIN]=60;

//   tcflush(m_fd, TCIFLUSH);
    if (tcsetattr(fd, TCSANOW, &newtio)!=0)
        printf("tsetaddr1 failed!\n");

    int mcs=0;

    ioctl(fd, TIOCMGET, &mcs);
    mcs |= TIOCM_RTS;
    ioctl(fd, TIOCMSET, &mcs);

    if (tcgetattr(fd, &newtio)!=0)
        printf("tcgetattr() 4 failed\n");

    newtio.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &newtio)!=0)
      printf("tcsetattr() 2 failed\n");

    return fd;
}

