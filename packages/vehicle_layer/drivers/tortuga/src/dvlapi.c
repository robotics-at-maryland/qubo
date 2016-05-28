/*
 * Copyright (C) 2009 Robotics at Maryland
 * Copyright (C) 2009 Kit Sczudlo <kitsczud@umd.edu>
 * All rights reserved.
 *
 * Author: Kit Sczudlo <kitsczud@umd.edu>
 * File:  packages/dvl/src/dvlapi.c
 */

// If you want to have the DVL API open a file instead of a tty,
// define this:
// #define DEBUG_DVL_OPEN_FILE

// STD Includes
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

// UNIX Includes
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

// Linux Includes (Not Sure If These Are needed)
#ifdef RAM_LINUX
  #include <bits/types.h>
  #include <linux/serial.h>
#endif // RAM_LINUX

// Project Includes
#include "dvlapi.h"

/* Takes two bytes and pops out a short */
uint16_t dvl_convert16(unsigned char msb, unsigned char lsb)
{
    return ((((uint16_t) msb) << 8) | lsb);
}

/* This receives a single byte */
unsigned char dvl_waitByte(int fd)
{
    unsigned char rxb;
    int temp;
    while((temp= read(fd, &rxb, 1)) != 1 && temp != EOF)
        ;

    return rxb;
}

/* This waits for the two starting bytes of 0x7f7f */
/* I have this failing after SYNC_FAIL_BYTECOUNT */
int dvl_waitSync(int fd)
{
    int syncLen= 0;

    while(syncLen < SYNC_FAIL_BYTECOUNT) {
        if(dvl_waitByte(fd) == 0x7D) {
            if(dvl_waitByte(fd) == 0x00)
                break;
         }

        syncLen++;
    }

    if(syncLen >= SYNC_FAIL_BYTECOUNT) {
        fprintf(stderr, "UNABLE TO SYCHRONIZE WITH DVL!\n");
        return -1;
    }

    return 0;
}

/* This reads in the data from the DVL and stores it so
   that the AI and controls guys have something to work with! */
int readDVLData(int fd, RawDVLData* dvl)
{
    /* So in the PD4 data format we should only get 47 bytes.
       We'll stick with the enormous buffer just in case.
       */
    unsigned char dvlData[512];

    int len, i, tempsize;
    uint16_t checksum;
    static CompleteDVLPacket dbgpkt;

    if(dvl_waitSync(fd))
        return dvl->valid= ERR_NOSYNC;

    dvl->valid= -1; // Packet is not yet valid
    dvl->privDbgInf= &dbgpkt;

    /* We got these in the dvl_waitSync() call */
    dvlData[0]= 0x7D;
    dvlData[1]= 0;

    /* Set length based on the 0x7D00 we recieved */
    len= 2;

    /* Get the packet size */
    while(len < 4)
        len+= read(fd, dvlData + len, 6 - len);

    tempsize= dvl_convert16(dvlData[3], dvlData[2]);

    while(len < tempsize)
        len+= read(fd, dvlData + len, tempsize - len);

    checksum= 0;
    for(i= 0;i < len;i++)
        checksum+= dvlData[i];

    tempsize+= 2;
    while(len < tempsize)
        len+= read(fd, dvlData + len, tempsize - len);

    dbgpkt.checksum= dvl_convert16(dvlData[46], dvlData[45]);

    if(checksum != dbgpkt.checksum) {
        fprintf(stderr, "WARNING! Bad checksum.\n");
        fprintf(stderr, "Expected 0x%02x but got 0x%02x\n", checksum, dbgpkt.checksum);
        dvl->valid= ERR_CHKSUM;
        return ERR_CHKSUM;
    }

    dvl->valid= 1;

    dvl->xvel_btm= dvl_convert16(dvlData[6], dvlData[5]);
    dvl->yvel_btm= dvl_convert16(dvlData[8], dvlData[7]);
    dvl->zvel_btm= dvl_convert16(dvlData[10], dvlData[9]);
    dvl->evel_btm= dvl_convert16(dvlData[12], dvlData[11]);

    dvl->beam1_range= dvl_convert16(dvlData[14], dvlData[13]);
    dvl->beam2_range= dvl_convert16(dvlData[16], dvlData[15]);
    dvl->beam3_range= dvl_convert16(dvlData[18], dvlData[17]);
    dvl->beam4_range= dvl_convert16(dvlData[20], dvlData[19]);

    dvl->TOFP_hundreths= dvlData[35];
    dvl->TOFP_hundreths*= 60;
    dvl->TOFP_hundreths+= dvlData[36];
    dvl->TOFP_hundreths*= 60;
    dvl->TOFP_hundreths+= dvlData[37];
    dvl->TOFP_hundreths*= 100;
    dvl->TOFP_hundreths+= dvlData[38];

    return 0;
}

/* Some code from cutecom, which in turn may have come from minicom */
int openDVL(const char* devName)
{
#ifdef DEBUG_DVL_OPEN_FILE
   int fd= openDVL(devName);

   return fd;
#else
   int fd = open(devName, O_RDWR, O_ASYNC);

    if(fd == -1)
        return -1;

    struct termios newtio;
    if (tcgetattr(fd, &newtio)!=0)
        printf("\nFirst stuff failed\n");

    unsigned int _baud= B115200;
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
#endif
}



/* We now need to trundle through lots of data, some of which we
       may not know how to parse. */
/*    while(len < dbgpkt->header.PacketSize - 4)
    {
        // The next chunk of data is two bytes which *should* identify the
        // next big chunk as being a Bottom Track packet.  Otherwise we're
        // FUCKED.
        tempsize+= 2;
        while(len < tempsize)
            len += read(fd, dvlData + len, tempsize - len);

        switch(dvl_convert16(dvlData[offset + 1], dvlData[offset]))
        {
            case(0x0600):
            {
                // This is a bottom track packet!

                // Read in all 79 bytes! (the id bytes were already read in!)
                tempsize+= 79;
                while(len < tempsize)
                    len += read(fd, dvlData + len, tempsize - len);

                dbgpkt->btdata.BottomTrackID= dvl_convert16(dvlData[offset + 1],
                                                        dvlData[offset]);
                dbgpkt->btdata.bt_pings_per_ensemble=
                    dvl_convert16(dvlData[offset + 3], dvlData[offset + 2]);
                dbgpkt->btdata.bt_delay_before_reaquire=
                    dvl_convert16(dvlData[offset + 5], dvlData[offset + 4]);
                dbgpkt->btdata.bt_corr_mag_min= dvlData[offset + 6];
                dbgpkt->btdata.bt_eval_amp_min= dvlData[offset + 7];
                dbgpkt->btdata.bt_prcnt_good_min= dvlData[offset + 8];
                dbgpkt->btdata.bt_mode= dvlData[offset + 9];
                dbgpkt->btdata.bt_err_vel= dvl_convert16(dvlData[offset + 11],
                                                     dvlData[offset + 10]);
                // Bytes 12 through 15 are empty!
                dbgpkt->btdata.bt_range[0]= dvl_convert16(dvlData[offset + 17],
                                                      dvlData[offset + 16]);
                dbgpkt->btdata.bt_range[1]= dvl_convert16(dvlData[offset + 19],
                                                      dvlData[offset + 18]);
                dbgpkt->btdata.bt_range[2]= dvl_convert16(dvlData[offset + 21],
                                                      dvlData[offset + 20]);
                dbgpkt->btdata.bt_range[3]= dvl_convert16(dvlData[offset + 23],
                                                      dvlData[offset + 22]);
                dbgpkt->btdata.bt_vel[0]= dvl_convert16(dvlData[offset + 25],
                                                    dvlData[offset + 24]);
                dbgpkt->btdata.bt_vel[1]= dvl_convert16(dvlData[offset + 27],
                                                    dvlData[offset + 26]);
                dbgpkt->btdata.bt_vel[2]= dvl_convert16(dvlData[offset + 29],
                                                    dvlData[offset + 28]);
                dbgpkt->btdata.bt_vel[3]= dvl_convert16(dvlData[offset + 31],
                                                    dvlData[offset + 30]);
                dbgpkt->btdata.bt_beam_corr[0]= dvlData[offset + 32];
                dbgpkt->btdata.bt_beam_corr[1]= dvlData[offset + 33];
                dbgpkt->btdata.bt_beam_corr[2]= dvlData[offset + 34];
                dbgpkt->btdata.bt_beam_corr[3]= dvlData[offset + 35];
                dbgpkt->btdata.bt_eval_amp[0]= dvlData[offset + 36];
                dbgpkt->btdata.bt_eval_amp[1]= dvlData[offset + 37];
                dbgpkt->btdata.bt_eval_amp[2]= dvlData[offset + 38];
                dbgpkt->btdata.bt_eval_amp[3]= dvlData[offset + 39];
                dbgpkt->btdata.bt_prcnt_good[0]= dvlData[offset + 40];
                dbgpkt->btdata.bt_prcnt_good[1]= dvlData[offset + 41];
                dbgpkt->btdata.bt_prcnt_good[2]= dvlData[offset + 42];
                dbgpkt->btdata.bt_prcnt_good[3]= dvlData[offset + 43];
                dbgpkt->btdata.ref_lyr_min= dvl_convert16(dvlData[offset + 45],
                                                      dvlData[offset + 44]);
                dbgpkt->btdata.ref_lyr_near= dvl_convert16(dvlData[offset + 47],
                                                       dvlData[offset + 46]);
                dbgpkt->btdata.ref_lyr_far= dvl_convert16(dvlData[offset + 49],
                                                      dvlData[offset + 48]);
                dbgpkt->btdata.ref_layer_vel[0]=
                    dvl_convert16(dvlData[offset + 51], dvlData[offset + 50]);
                dbgpkt->btdata.ref_layer_vel[1]=
                    dvl_convert16(dvlData[offset + 53], dvlData[offset + 52]);
                dbgpkt->btdata.ref_layer_vel[2]=
                    dvl_convert16(dvlData[offset + 55], dvlData[offset + 54]);
                dbgpkt->btdata.ref_layer_vel[3]=
                    dvl_convert16(dvlData[offset + 57], dvlData[offset + 56]);
                dbgpkt->btdata.ref_corr[0]= dvlData[offset + 58];
                dbgpkt->btdata.ref_corr[1]= dvlData[offset + 59];
                dbgpkt->btdata.ref_corr[2]= dvlData[offset + 60];
                dbgpkt->btdata.ref_corr[3]= dvlData[offset + 61];
                dbgpkt->btdata.ref_int[0]= dvlData[offset + 62];
                dbgpkt->btdata.ref_int[1]= dvlData[offset + 63];
                dbgpkt->btdata.ref_int[2]= dvlData[offset + 64];
                dbgpkt->btdata.ref_int[3]= dvlData[offset + 65];
                dbgpkt->btdata.ref_prcnt_good[0]= dvlData[offset + 66];
                dbgpkt->btdata.ref_prcnt_good[1]= dvlData[offset + 67];
                dbgpkt->btdata.ref_prcnt_good[2]= dvlData[offset + 68];
                dbgpkt->btdata.ref_prcnt_good[3]= dvlData[offset + 69];
                dbgpkt->btdata.bt_max_depth= dvl_convert16(dvlData[offset + 71],
                                                       dvlData[offset + 70]);
                dbgpkt->btdata.rssi_amp[0]= dvlData[offset + 72];
                dbgpkt->btdata.rssi_amp[1]= dvlData[offset + 73];
                dbgpkt->btdata.rssi_amp[2]= dvlData[offset + 74];
                dbgpkt->btdata.rssi_amp[3]= dvlData[offset + 75];
                dbgpkt->btdata.gain= dvlData[offset + 76];
                dbgpkt->btdata.bt_range_msb[0]= dvlData[offset + 77];
                dbgpkt->btdata.bt_range_msb[1]= dvlData[offset + 78];
                dbgpkt->btdata.bt_range_msb[2]= dvlData[offset + 79];
                dbgpkt->btdata.bt_range_msb[3]= dvlData[offset + 80];

                break;
            }

            default:
            {
                // If we're here we don't understand the data being sent!
                // we'll discard it using the offset information and move
                // along!

                printf("WARNING! Unknown Datatype!\n");

                // Find the offset
                for(i= 0;i < dbgpkt->header.num_datatypes &&
                         dbgpkt->header.offsets[i] < tempsize;i++)
                    ;

                // Make it the new size
                if(i == dbgpkt->header.num_datatypes) {
                    tempsize+= (dbgpkt->header.PacketSize - 4) -
                               (dbgpkt->header.offsets[i - 1]);
                } else {
                    tempsize+= dbgpkt->header.offsets[i] -
                               dbgpkt->header.offsets[i - 1];
                }

                // Read it into the buffer in order to eat it!
                while(len < tempsize)
                    len += read(fd, dvlData + len, tempsize - len);

                break;
            }

        }

        // Keep track of the new offset!
        offset= tempsize;
    } */
