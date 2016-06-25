/*
 * Copyright (C) 2011 Robotics at Maryland
 * Copyright (C) 2011 Kit Sczudlo <kitsczud@umd.edu>
 * All rights reserved.
 *
 * Author: Kit Sczudlo <kitsczud@umd.edu>
 * File:  packages/dvl/include/dvlapi.h
 */

#ifndef RAM_DVLAPI_H_06_18_2011
#define RAM_DVLAPI_H_06_18_2011

// If we are compiling as C++ code we need to use extern "C" linkage
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdint.h>

/* We should abort trying to sync if it takes more than a
   couple packets worth of bytes, or more than 1 second. */
#define SYNC_FAIL_BYTECOUNT 2000
#define SYNC_FAIL_MILLISEC 1000
#define SYNC_FAIL_SECONDS (SYNC_FAIL_MILLISEC/1000)

/* These are error messages */
#define ERR_NOSYNC            0x0001
#define ERR_TOOBIG            0x0002
#define ERR_CHKSUM            0x0006

/* This will hold *ALL* of the data from the DVL */
/* It should NOT be passed every time the sensor is polled */
typedef struct _CompleteDVLPacket
{
    uint8_t sysconf;

    int16_t xvel_btm;
    int16_t yvel_btm;
    int16_t zvel_btm;
    int16_t evel_btm;

    uint16_t beam1_range;
    uint16_t beam2_range;
    uint16_t beam3_range;
    uint16_t beam4_range;

    uint8_t btm_status;

    int16_t xvel_ref;
    int16_t yvel_ref;
    int16_t zvel_ref;
    int16_t evel_ref;

    uint16_t ref_lyr_start;
    uint16_t ref_lyr_end;
    uint8_t ref_lyr_status;

    // TOFP stands for Time Of First Ping
    uint8_t TOFP_hour;
    uint8_t TOFP_min;
    uint8_t TOFP_sec;
    uint8_t TOFP_hnd;

    // BIT stands for Built-In Tests
    uint16_t BIT_results;

    uint16_t speed_of_sound;

    int16_t temperature;

    uint16_t checksum;
} CompleteDVLPacket;

/* This is the info that will actually get passed to and from
   the API. */
typedef struct _RawDVLData
{
    /* Useful information in this structure? */
    /* Non-zero implies invalid data. */
    unsigned int valid;

    /* vvvvv PUT DATA HERE vvvvv */
    int16_t xvel_btm;
    int16_t yvel_btm;
    int16_t zvel_btm;
    int16_t evel_btm;

    uint16_t beam1_range;
    uint16_t beam2_range;
    uint16_t beam3_range;
    uint16_t beam4_range;

    // TOFP stands for Time Of First Ping
    // this is the hundredths
    uint32_t TOFP_hundreths;
    /* ^^^^^ PUT DATA HERE ^^^^^ */

    CompleteDVLPacket *privDbgInf;
} RawDVLData;

/** Opens a serial channel to the imu using the given devices
 *
 *  @param  devName  Device filename
 *
 *  @return  The file descriptor of the device file, -1 on error.
 */
int openDVL(const char *devName);

/** Read the latest DVL measurements into the given structure
 *
 *  @param fd  The device file returned by openDVL
 */
int readDVLData(int fd, RawDVLData *dvl);

// If we are compiling as C++ code we need to use extern "C" linkage
#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
    
#endif // RAM_DVLAPI_H_06_18_2011
