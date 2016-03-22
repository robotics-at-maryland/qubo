/*
 * Copyright (C) 2007 Robotics at Maryland
 * Copyright (C) 2007 Steve Moskovchenko <stevenm@umd.edu>
 * All rights reserved.
 *
 * Author: Steve Moskovchenko <stevenm@umd.edu>
 * File:  packages/sensorapi/src/sensorapi.c
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
#include "sensorapi.h"

#define USEC_PER_MSEC 1000

#ifndef SENSORAPI_R5
#error "WRONG VERSION OF INCLUDE"
#endif

#include <math.h>

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
        return SB_IOERROR;

    return read(fd, buf, nbytes);
}

int syncBoard(int fd)
{
    unsigned char buf[5];

    /* Eat the incoming buffer */
    while(hasData(fd, 0))
    {
      if (1 != read(fd, buf, 1))
	return SB_ERROR;
    }

    int i;

    for(i=0; i<MAX_SYNC_ATTEMPTS; i++)
    {
        unsigned char b[1];
        b[0] = 0xFF;
        writeData(fd, b, 1);
	    miniSleep();

        b[0]=0;

        if(hasData(fd, IO_TIMEOUT))
            readData(fd, b, 1);
	    miniSleep();

        if(!hasData(fd, IO_TIMEOUT) && b[0]==0xBC)
            return SB_OK;
    }

    return SB_ERROR;
}


int pingBoard(int fd)
{
    unsigned char buf[2]={HOST_CMD_PING, HOST_CMD_PING};
    writeData(fd, buf, 2);
    readData(fd, buf, 1);
    if(buf[0] == 0xBC)
        return SB_OK;
    return SB_HWFAIL;
}


int checkBoard(int fd)
{
    unsigned char buf[2]={HOST_CMD_SYSCHECK, HOST_CMD_SYSCHECK};
    writeData(fd, buf, 2);
    readData(fd, buf, 1);
    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    return SB_ERROR;
}




int readDepth(int fd)
{
    unsigned char buf[5]={HOST_CMD_DEPTH, HOST_CMD_DEPTH};
    writeData(fd, buf, 2);
    readData(fd, buf, 1);
    if(buf[0] != HOST_REPLY_DEPTH)
        return SB_ERROR;

    readData(fd, buf, 3);

    if( ((0x03 + buf[0] + buf[1]) & 0xFF) == buf[2])
        return (buf[0]<<8 | buf[1]);

    return SB_ERROR;
}


/*
 * Send:   [CmdCode, CS]
 * Expect: [ReplyCode, Val, CS]
 */
int simpleRead(int fd, int cmdCode, int replyCode)
{
    unsigned char buf[5];
    buf[0] = buf[1] = cmdCode;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != replyCode)
    {
        printf("Bad reply from simple command %02x! (Expected %02x, got %02x)\n", cmdCode, replyCode, buf[0]);
        return SB_ERROR;
    }
    readData(fd, buf, 2);

    if( ((replyCode + buf[0]) & 0xFF) == buf[1])
        return buf[0];

    printf("Bad cs in response from simple command %02x!\n", cmdCode);

    return SB_ERROR;
}


/* The pre-2-byte status reply
int readStatus(int fd)
{
    return simpleRead(fd, HOST_CMD_BOARDSTATUS, HOST_REPLY_BOARDSTATUS);
}*/

int readStatus(int fd)
{
    int ret= 0;
    unsigned char buf[4];
    buf[0]= buf[1]= HOST_CMD_BOARDSTATUS;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != HOST_REPLY_BOARDSTATUS) {
        printf("Bad reply while attempting to recieve status! \
                (Expected %02x, got %02x)\n", HOST_REPLY_BOARDSTATUS, buf[0]);
        return SB_ERROR;
    }

    readData(fd, buf, 3);
    if(buf[2] != ((buf[0]+buf[1]+HOST_REPLY_BOARDSTATUS) & 0xFF)) {
        printf("Bad checksum while recieving status!\n");
        printf("Got %x but was expecting %x\n", buf[2], ((buf[0] + buf[1] + HOST_REPLY_BOARDSTATUS) & 0xFF));
        printf("%x; %x; %x\n", buf[0], buf[1], HOST_REPLY_BOARDSTATUS);
        return SB_ERROR;
    }

    /* Now we store the two bytes in one int. */
    ret|= buf[0];
    ret<<= 8;
    ret|= buf[1];

    return ret;
}


int readThrusterState(int fd)
{
    return simpleRead(fd, HOST_CMD_THRUSTERSTATE, HOST_REPLY_THRUSTERSTATE);
}


int readBatteryEnables(int fd)
{
    return simpleRead(fd, HOST_CMD_BATTSTATE, HOST_REPLY_BATTSTATE);
}

int readBatteryUsage(int fd)
{
    return readStatus(fd) & 0x3F;
}


int readBarState(int fd)
{
    return simpleRead(fd, HOST_CMD_BARSTATE, HOST_REPLY_BARSTATE);
}

int readOvrState(int fd)
{
    return simpleRead(fd, HOST_CMD_READ_OVR, HOST_REPLY_OVR);
}


int readTemp(int fd, unsigned char * tempData)
{
    unsigned char buf[10]={HOST_CMD_TEMPERATURE, HOST_CMD_TEMPERATURE};
    int i;
    for(i=0; i<NUM_TEMP_SENSORS; i++)
        tempData[i]=0;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);



    if(buf[0] != 0x0B)
        return SB_ERROR;

    readData(fd, tempData, NUM_TEMP_SENSORS);
    readData(fd, buf, 1);

    unsigned char sum = 0x0B;

    for(i=0; i<NUM_TEMP_SENSORS; i++)
        sum = (sum+tempData[i]) & 0xFF;

    if(sum == buf[0])
        return SB_OK;

    return SB_ERROR;
}


int getSonarData(int fd, struct sonarData * sd)
{
    unsigned char buf[22]={HOST_CMD_SONAR, HOST_CMD_SONAR};
    int i;
    unsigned char rawSonar[22] = {0,0,0,0,0};

    if(sd == NULL)
        return -1;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != 0x0E)
        return SB_ERROR;

    readData(fd, rawSonar, 22);
    readData(fd, buf, 1);

    unsigned char sum = 0x0E;

    for(i=0; i<22; i++)
        sum = (sum+rawSonar[i]) & 0xFF;

    if(sum != buf[0])
        return SB_ERROR;

/*    printf("\nDebug: Received data from sonar board: < ");
    for(i=0; i<20; i++)
        printf("0x%02X ", rawSonar[i]);

    printf(">\n");

    printf("%d\n", ((rawSonar[0]<<8) | rawSonar[1]));
*/
    int errorCount = 0;

    if(rawSonar[0] != 0x00)
        errorCount++;

    sd->vectorX = ((signed short) ((rawSonar[1]<<8) | rawSonar[2])) / 10000.0;
    sd->vectorY = ((signed short) ((rawSonar[3]<<8) | rawSonar[4])) / 10000.0;
    sd->status = rawSonar[5];

    if(rawSonar[6] != 0x00)
        errorCount++;

    sd->vectorZ = ((signed short) ((rawSonar[7]<<8) | rawSonar[8])) / 10000.0;

    sd->range = (rawSonar[9]<<8) | rawSonar[10];

    if(rawSonar[11] != 0x00)
        errorCount++;

    sd->timeStampSec = (rawSonar[12]<<24) | (rawSonar[13] << 16) | (rawSonar[14] << 8) | rawSonar[15];

    if(rawSonar[16] != 0x00)
        errorCount++;

    sd->timeStampUSec = (rawSonar[17]<<24) | (rawSonar[18] << 16) | (rawSonar[19] << 8) | rawSonar[20];


    /// TODO: Put which pinger we are going for in the message
    sd->pingerID = 0;
    
    unsigned char cs = 0;

    for(i=0; i<21; i++)
        cs += rawSonar[i];

    if(rawSonar[21] != cs)
        printf("Sonar checksum mismatch: expected %02x, got %02x\n", rawSonar[21], cs);

    if(errorCount)
    {
        printf("Sonar: %d bytes messed up\n", errorCount);
        printf("\nDebug: Received data from sonar board: < ");
        for(i=0; i<20; i++)
            printf("0x%02X ", rawSonar[i]);

        printf(">\n");
    }

    //printf("Vector: \t<%5.4f %5.4f %5.4f>\n", sd->vectorX, sd->vectorY, sd->vectorZ);
    //printf("Status: \t0x%02x\n", sd->status);
    //printf("Range:  \t%u\n", sd->range);
    //printf("Timestamp:\t%u\n", sd->timeStampSec);
    //printf("Sample No:\t%u\n", sd->timeStampUSec);
    //printf("Yaw: %f\n", atan2(sd->vectorY, sd->vectorX) * 180.0 / 3.14159);

    return SB_OK;
}


int hardKill(int fd)
{
    unsigned char buf[6]={0x06, 0xDE, 0xAD, 0xBE, 0xEF, 0x3E};
    writeData(fd, buf, 6);
    readData(fd, buf, 1);

    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    return SB_ERROR;
}

/*  Send:   [cmdCode, param, CS]
 *  Expect: [BC | DF | CC]
 *  Param valid is from 0 to range-1, inclusive
 */
int simpleWrite(int fd, int cmdCode, int param, int range)
{
    if(param < 0 || param > range)
        return -255;

    unsigned char buf[3];
    buf[0] = cmdCode;
    buf[1] = param;
    buf[2] = (cmdCode + param) & 0xFF;

    writeData(fd, buf, 3);
    readData(fd, buf, 1);

    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    return SB_ERROR;
}

int resetBlackfin(int fd)
{
    unsigned char buf[2]={HOST_CMD_BFRESET, HOST_CMD_BFRESET};
    writeData(fd, buf, 2);
    readData(fd, buf, 1);
    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    return SB_ERROR;
}

int startBlackfin(int fd)
{
    return simpleWrite(fd, HOST_CMD_BFIN_STATE, 1, 2);
}

int stopBlackfin(int fd)
{
    return simpleWrite(fd, HOST_CMD_BFIN_STATE, 0, 2);
}

int switchToExternalPower(int fd)
{
    return simpleWrite(fd, HOST_CMD_SWITCHPOWER, 1, 2);
}

int switchToInternalPower(int fd)
{
    return simpleWrite(fd, HOST_CMD_SWITCHPOWER, 0, 2);
}

int setBatteryState(int fd, int state)
{
    return simpleWrite(fd, HOST_CMD_BATTCTL, state, 10);
}

int dropMarker(int fd, int markerNum)
{
    return simpleWrite(fd, HOST_CMD_MARKER, markerNum, 2);
}

int lcdBacklight(int fd, int state)
{
    return simpleWrite(fd, HOST_CMD_BACKLIGHT, state, 3);
}

int setThrusterSafety(int fd, int state)
{
    if(state<0 || state>11)
        return -255;

    unsigned char buf[8]={0x09, 0xB1, 0xD0, 0x23, 0x7A, 0x69, 0, 0};

    buf[6] = state;

    int i;

    for(i=0; i<7; i++)
        buf[7] += buf[i];


    if(state > 5)	/* If unsafing, sleep a little */
    	usleep(300 * 1000);


    writeData(fd, buf, 8);

    readData(fd, buf, 1);

    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    return SB_ERROR;
}


int setBarState(int fd, int state)
{
    return simpleWrite(fd, HOST_CMD_BARS, state, 16);
}

int setAnimation(int fd, int anim)
{
    return simpleWrite(fd, HOST_CMD_BARANIMATION, anim, 3);
}

int setBarOutputs(int fd, int bars)
{
    return simpleWrite(fd, HOST_CMD_SET_BARS, bars, 256);
}

int displayText(int fd, int line, const char* text)
{
    if(line!=0 && line!=1)
        return -255;

    if(!text)
        return 0;

    unsigned char buf[20]={0x0C, 0,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,0 };

    buf[1] = line;

    int i;

    for(i=0; text[i]!=0 && i<16; i++)
        buf[i+2]=text[i];

    for(i=0; i<18; i++)
        buf[18] += buf[i];


    writeData(fd, buf, 19);

    readData(fd, buf, 1);

    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    return SB_ERROR;
}


// MSB LSB !! (big endian)
int setSpeeds(int fd, int s1, int s2, int s3, int s4, int s5, int s6)
{
    int i=0;
    unsigned char buf[14]={HOST_CMD_SETSPEED, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0x00};
//    printf("Sending speeds: %d %d %d %d\n", s1, s2, s3, s4);


    buf[1] = (s1 >> 8);
    buf[2] = (s1 & 0xFF);

    buf[3] = (s2 >> 8);
    buf[4] = (s2 & 0xFF);

    buf[5] = (s3 >> 8);
    buf[6] = (s3 & 0xFF);

    buf[7] = (s4 >> 8);
    buf[8] = (s4 & 0xFF);

    buf[9] = (s5 >> 8);
    buf[10] = (s5 & 0xFF);

    buf[11] = (s6 >> 8);
    buf[12] = (s6 & 0xFF);

    buf[13] = 0;

    for(i=0; i<13; i++)
        buf[13]+=buf[i];

    writeData(fd, buf, 14);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

// 14 xx xx xx xx CS
int readSpeedResponses(int fd)
{
    unsigned char buf[8]={0x13, 0x13};
    int i;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != 0x14)
    {
    	printf("Bad reply: %x\n", buf[0]);
    	return SB_ERROR;
    }
    readData(fd, buf+1, 7);

    unsigned char sum = 0;

    for(i=0; i<7; i++)
        sum = (sum+buf[i]) & 0xFF;

    if(sum != buf[7])
    {
        printf("bad cs: got %02x, expected %02x\n", sum, buf[7]);
        return SB_ERROR;
    }

    int errCount=0;

    for(i=0; i<6; i++)
        if(buf[i+1] != 0x06)
            errCount++;

    if(errCount != 0)
    {
        printf("\t Got: %02x %02x %02x %02x %02x %02x\n", buf[1], buf[2], buf[3], buf[4], buf[5], buf[6]);
        return SB_ERROR;
    }

    return SB_OK;

}

int readMotorCurrents(int fd, struct powerInfo * info)
{
    unsigned char buf[20] = {HOST_CMD_IMOTOR, HOST_CMD_IMOTOR};
    int i=0, cs=0;

    if(info == NULL)
        return SB_ERROR;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != HOST_REPLY_IMOTOR)
    {
        if(buf[0] == 0xCC)
            return SB_BADCC;
        if(buf[0] == 0xDF)
            return SB_HWFAIL;

        return SB_ERROR;
    }

    readData(fd, buf+1, 17);

    for(i=0; i<17; i++)
        cs += buf[i];

    if((cs & 0xFF) != buf[17])
        return SB_BADCC;

    for(i=0; i<8; i++)
        info->motorCurrents[i] = ((buf[i*2+1] << 8) | (buf[i*2+2])) / 1000.0;

    return SB_OK;
}

int readBoardVoltages(int fd, struct powerInfo * info)
{
    unsigned char buf[20] = {HOST_CMD_VLOW, HOST_CMD_VLOW};
    int i=0, cs=0;

    if(info == NULL)
        return SB_ERROR;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != HOST_REPLY_VLOW)
    {
        if(buf[0] == 0xCC)
            return SB_BADCC;
        if(buf[0] == 0xDF)
            return SB_HWFAIL;

        return SB_ERROR;
    }

    readData(fd, buf+1, 11);


    for(i=0; i<11; i++)
        cs += buf[i];

    if((cs & 0xFF) != buf[11])
        return SB_BADCC;


    info->v5VBus = ((buf[0*2+1] << 8) | (buf[0*2+2])) / 1000.0;
    info->i5VBus = ((buf[1*2+1] << 8) | (buf[1*2+2])) / 1000.0;
    info->v12VBus = ((buf[2*2+1] << 8) | (buf[2*2+2])) / 1000.0;
    info->i12VBus = ((buf[3*2+1] << 8) | (buf[3*2+2])) / 1000.0;
    info->iAux = ((buf[4*2+1] << 8) | (buf[4*2+2])) / 1000.0;
    return SB_OK;
}



int readBatteryVoltages(int fd, struct powerInfo * info)
{
    unsigned char buf[18] = {HOST_CMD_BATTVOLTAGE, HOST_CMD_BATTVOLTAGE};
    int i=0, cs=0;

    if(info == NULL)
        return SB_ERROR;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != HOST_REPLY_BATTVOLTAGE)
    {

        printf("\nbad reply!\n");

        if(buf[0] == 0xCC)
            return SB_BADCC;
        if(buf[0] == 0xDF)
            return SB_HWFAIL;

        return SB_ERROR;
    }

    readData(fd, buf+1, 15);

    for(i=0; i<15; i++)
        cs += buf[i];

    if((cs & 0xFF) != buf[15])
    {
        printf("bad cs in voltages!\n");
        return SB_BADCC;
    }

    for(i=0; i<6; i++)
        info->battVoltages[i] = ((buf[i*2+1] << 8) | (buf[i*2+2])) / 1000.0;

    info->v26VBus = ((buf[6*2+1] << 8) | (buf[6*2+2])) / 1000.0;

    return SB_OK;
}

int readBatteryCurrents(int fd, struct powerInfo * info)
{
    unsigned char buf[14] = {HOST_CMD_BATTCURRENT, HOST_CMD_BATTCURRENT};
    int i=0, cs=0;

    if(info == NULL)
        return SB_ERROR;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != HOST_REPLY_BATTCURRENT)
    {

        printf("\nbad reply!\n");

        if(buf[0] == 0xCC)
            return SB_BADCC;
        if(buf[0] == 0xDF)
            return SB_HWFAIL;

        return SB_ERROR;
    }

    readData(fd, buf+1, 13);

    for(i=0; i<13; i++)
        cs += buf[i];

    if((cs & 0xFF) != buf[13])
    {
        printf("bad cc in currents!\n");
        return SB_BADCC;
    }

    for(i=0; i<6; i++)
        info->battCurrents[i] = ((buf[i*2+1] << 8) | (buf[i*2+2])) / 1000.0;

    return SB_OK;
}


/* Runtime diagnostics not really available in SB R5 due to the massive number of LEDs */
int setDiagnostics(int fd, int state)
{
    return simpleWrite(fd, HOST_CMD_RUNTIMEDIAG, state, 2);
}


int readOvrParams(int fd, int * a, int * b)
{
    unsigned char buf[4];

    if(a == NULL || b == NULL)
        return SB_ERROR;

    buf[0] = HOST_CMD_READ_OVRLIMIT;
    buf[1] = buf[0];

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] != HOST_REPLY_OVRLIMIT)
        return SB_ERROR;

    readData(fd, buf+1, 3);

    if(((buf[0] + buf[1] + buf[2]) & 0xFF) != buf[3])
	{
		printf("got cs: %02x, read: %02x\n", (buf[0] + buf[1] + buf[2]) & 0xFF, buf[3]);
        return SB_BADCC;
	}
    *a = buf[1];
    *b = buf[2];

    return SB_OK;
}


int setOvrParams(int fd, int a, int b)
{
    if(a < 0 || a > 255 || b < 0 || b > 255)
        return -255;

    unsigned char buf[4];
    buf[0] = HOST_CMD_SET_OVRLIMIT;
    buf[1] = a;
	buf[2] = b;
    buf[3] = buf[0] + buf[1] + buf[2];

    writeData(fd, buf, 4);
    readData(fd, buf, 1);

    if(buf[0] == 0xBC)
        return SB_OK;

    if(buf[0] == 0xCC)
        return SB_BADCC;

    if(buf[0] == 0xDF)
        return SB_HWFAIL;

    return SB_ERROR;
}


int partialRead(int fd, struct boardInfo * info)
{
    int retCode=0;
    if(info == NULL)
        return SB_ERROR;

    // Increment to find out what we are updating this time
    info->updateState++;

    // Roll over based upon end of enum marker
    if (END_OF_UPDATES == info->updateState)
        info->updateState = STATUS;

    switch(info->updateState)
    {
        // Note STATUS == 1
        case STATUS:
        {
            info->status = retCode = readStatus(fd);
            break;
        }

        case THRUSTER_STATE:
        {
            info->thrusterState = retCode = readThrusterState(fd);
            break;
        }

        case BAR_STATE:
        {
            info->barState = retCode = readBarState(fd);
            break;
        }

        case OVERCURRENT_STATE:
        {
            info->ovrState = retCode = readOvrState(fd);
            break;
        }

        case BATTERY_ENABLES:
        {
            info->battEnabled = retCode = readBatteryEnables(fd);
            break;
        }

        case TEMP:
        {
            retCode = readTemp(fd, info->temperature);
            break;
        }

        case MOTOR_CURRENTS:
        {
            retCode = readMotorCurrents(fd, &(info->powerInfo));
            break;
        }

        case BOARD_VOLTAGES_CURRENTS:
        {
            retCode = readBoardVoltages(fd, &(info->powerInfo));
            break;
        }

        case BATTERY_VOLTAGES:
        {
            retCode = readBatteryVoltages(fd, &(info->powerInfo));
            break;
        }

        case BATTERY_CURRENTS:
        {
            retCode = readBatteryCurrents(fd, &(info->powerInfo));
            break;
        }

        case BATTERY_USED:
        {
            info->battUsed = retCode = readBatteryUsage(fd);
            break;
        }

        case SONAR:
        {
            retCode = getSonarData(fd, &(info->sonar));
            break;
        }

        default:
        {
            printf("ERROR: update rolled over");
            info->updateState = STATUS;
            return SB_OK;
        }

    }

    // If we just updated the last item, we have finished an update cycle
    if((END_OF_UPDATES - 1) == info->updateState)
    {
        if(retCode >= 0)
            return SB_UPDATEDONE;
        else
            return retCode;
    }

    if(retCode < 0)
        return retCode;
    else
        return SB_OK;
}


/* Some code from cutecom, which in turn may have come from minicom */
/* FUGLY but it does what I want */
int openSensorBoard(const char * devName)
{
   int fd = open(devName, O_RDWR, O_ASYNC); // | O_ASYNC); //, O_RDWR, O_NONBLOCK);

    if(fd == -1)
        return -1;

    struct termios newtio;
    if (tcgetattr(fd, &newtio)!=0)
        printf("\nFirst stuff failed\n");

    unsigned int _baud=B115200;
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

char* sbErrorToText(int ret)
{
    static char* toText[] = {
        "IO error",
        "Bad CRC",
        "Hardware failure",
        "Error",
        "OK"
    };

    if ((ret >= -4) && (ret <= 0))
        return toText[ret + 4];
    else
        return "Unknown";
}

char* tempSensorIDToText(int id)
{
    static char* toText[7] = {
        "Sensor Board",
        "Unused",
        "Unused",
        "Unused",
        "Unused",
        "Distro Board",
        "Balancer Board"
    };

    if ((id >=0) && (id <= (sizeof(toText))))
        return toText[id];
    else
        return "ERROR: Id out of range";
}

int setServoPower(int fd, unsigned char power)
{
    /* The command has been removed, see "setMagPower" */
    /*
    unsigned char buf[2];
    if(power) {
        buf[0]= buf[1]= HOST_CMD_SERVO_POWER_ON;
    } else {
        buf[0]= buf[1]= HOST_CMD_SERVO_POWER_OFF;
    }
    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;*/

    return SB_ERROR;
}

int setMagPower(int fd, unsigned char power)
{
    /*
    unsigned char buf[2];
    if(power) {
        buf[0]= buf[1]= HOST_CMD_MAG_PWR_ON;
    } else {
        buf[0]= buf[1]= HOST_CMD_MAG_PWR_OFF;
    }
    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;
    */

    return SB_ERROR;
}

int setServoEnable(int fd, unsigned char servoMask)
{
    /* The command is depreciated. Do not use it. */
    return SB_ERROR;
    /*return simpleWrite(fd, HOST_CMD_SERVO_ENABLE, servoMask, 0x100);*/
}

int setServoPosition(int fd, unsigned char servoNumber, unsigned short position)
{
    /* This command is depreciated, do not use it. */
    /*
    unsigned int tmpchksum= 0;
    unsigned char buf[5]= { HOST_CMD_SET_SERVO_POS, servoNumber, (position >> 8) & 0xFF, position & 0xFF, 0 };

    tmpchksum+= buf[0];
    tmpchksum+= buf[1];
    tmpchksum+= buf[2];
    tmpchksum+= buf[3];
    buf[4]= (tmpchksum & 0xFF);

    writeData(fd, buf, 5);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;*/

    return SB_ERROR;
}

int DVLOn(int fd, unsigned char power)
{
    unsigned char buf[2];
    if(power) {
        buf[0]= buf[1]= HOST_CMD_DVL_ON;
    } else {
        buf[0]= buf[1]= HOST_CMD_DVL_OFF;
    }
    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int fireTorpedo(int fd, unsigned char torpnum)
{
    unsigned char buf[2];
    if(torpnum == 1) {
        buf[0]= buf[1]= HOST_CMD_FIRE_TORP_1;
    } else if(torpnum == 2) {
        buf[0]= buf[1]= HOST_CMD_FIRE_TORP_2;
    } else {
        return SB_ERROR;
    }

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int voidTorpedo(int fd, unsigned char torpnum)
{
    unsigned char buf[2];
    if(torpnum == 1) {
        buf[0]= buf[1]= HOST_CMD_VOID_TORP_1;
    } else if(torpnum == 2) {
        buf[0]= buf[1]= HOST_CMD_VOID_TORP_2;
    } else {
        return SB_ERROR;
    }

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int armTorpedo(int fd, unsigned char torpnum)
{
    unsigned char buf[2];
    if(torpnum == 1) {
        buf[0]= buf[1]= HOST_CMD_ARM_TORP_1;
    } else if(torpnum == 2) {
        buf[0]= buf[1]= HOST_CMD_ARM_TORP_2;
    } else {
        return SB_ERROR;
    }

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

//Kanga - Allowing for separate grabber extension
int extendGrabber(int fd, int param)
{
    unsigned char buf[2];

    switch(param)
    {
        case 0: 
        {
            buf[0]= buf[1]= HOST_CMD_EXT_GRABBER; 
            break;
        }
        case 1: 
        {
            buf[0]= buf[1]= HOST_CMD_EXT_GRABBER_1; 
            break;
        }
        case 2: 
        {
            buf[0]= buf[1]= HOST_CMD_EXT_GRABBER_2; 
            break;
        }
    }

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int retractGrabber(int fd)
{
    unsigned char buf[2];

    buf[0]= buf[1]= HOST_CMD_RET_GRABBER;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int voidGrabber(int fd)
{
    unsigned char buf[2];

    buf[0]= buf[1]= HOST_CMD_VOID_GRABBER;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int voidSystem(int fd)
{
    unsigned char buf[2];

    buf[0]= buf[1]= HOST_CMD_VOID_PNEU;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int pneumaticsOff(int fd)
{
    unsigned char buf[2];

    buf[0]= buf[1]= HOST_CMD_OFF_PNEU;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

/* if (on == 1) it will turn derpy on, otherwise it will turn derpy off */
int setDerpyPower(int fd, unsigned char on)
{
    unsigned char buf[2];

    if(on == 1) {
        buf[0]= buf[1]= HOST_CMD_DERPY_ON;
    } else {
        buf[0]= buf[1]= HOST_CMD_DERPY_OFF;
    }

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int setDerpySpeed(int fd, int speed)
{
    int i;
    unsigned char buf[8];

    buf[0]= HOST_CMD_SET_DERPY;
    buf[1]= (speed >> 8);
    buf[2]= (speed & 0xff);
    buf[3]= 'D';
    buf[4]= 'E';
    buf[5]= 'R';
    buf[6]= 'P';
    buf[7]= 0; // zero out the checksum

    for(i= 0;i < 7;++i)
        buf[7]+= buf[i];

    writeData(fd, buf, 8);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int stopDerpy(int fd)
{
    unsigned char buf[2];

    buf[0]= buf[1]= HOST_CMD_STOP_DERPY;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

/*Turn the camera connection on or off.
  kanga 7/2/2013*/

int camConnect(int fd){
    unsigned char buf [2];

    buf[0] = buf[1] = HOST_CMD_CAM_RELAY_ON;
    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
}

int camDisconnect(int fd){
    unsigned char buf[2];
    
    buf[0]=buf[1]= HOST_CMD_CAM_RELAY_OFF;

    writeData(fd, buf, 2);
    readData(fd, buf, 1);

    if(buf[0] == HOST_REPLY_SUCCESS)
        return SB_OK;

    if(buf[0] == HOST_REPLY_BADCHKSUM)
        return SB_BADCC;

    if(buf[0] == HOST_REPLY_FAILURE)
        return SB_HWFAIL;

    return SB_ERROR;
} 
