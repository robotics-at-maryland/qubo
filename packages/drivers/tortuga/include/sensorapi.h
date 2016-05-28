/*
 * Copyright (C) 2008 Robotics at Maryland
 * Copyright (C) 2008 Steve Moskovchenko <stevenm@umd.edu>
 * All rights reserved.
 *
 * Author: Steve Moskovchenko <stevenm@umd.edu>
 * File:  packages/sensorapi/src/sensorapi.c
 */

#ifndef RAM_DRIVER_SENSORAPI_H_06_09_2008
#define RAM_DRIVER_SENSORAPI_H_06_09_2008

#include "buscodes.h"

#define MAX_SYNC_ATTEMPTS 20
#define NUM_TEMP_SENSORS 7

#define SENSORAPI_R5

/* In msec */
#define IO_TIMEOUT  100

/* LCD backlight control */
#define LCD_BL_OFF    0
#define LCD_BL_ON     1
#define LCD_BL_FLASH  2

/* Control command return values */
#define SB_OK        0
#define SB_UPDATEDONE 1
#define SB_IOERROR  -4
#define SB_BADCC    -3
#define SB_HWFAIL   -2
#define SB_ERROR    -1


/* Inputs to the thruster safety command */
#define CMD_THRUSTER1_OFF     0
#define CMD_THRUSTER2_OFF     1
#define CMD_THRUSTER3_OFF     2
#define CMD_THRUSTER4_OFF     3
#define CMD_THRUSTER5_OFF     4
#define CMD_THRUSTER6_OFF     5

#define CMD_THRUSTER1_ON      6
#define CMD_THRUSTER2_ON      7
#define CMD_THRUSTER3_ON      8
#define CMD_THRUSTER4_ON      9
#define CMD_THRUSTER5_ON      10
#define CMD_THRUSTER6_ON      11



/* Inputs to the bar command */
#define CMD_BAR1_OFF     0x00
#define CMD_BAR2_OFF     0x01
#define CMD_BAR3_OFF     0x02
#define CMD_BAR4_OFF     0x03
#define CMD_BAR5_OFF     0x04
#define CMD_BAR6_OFF     0x05
#define CMD_BAR7_OFF     0x06
#define CMD_BAR8_OFF     0x07

#define CMD_BAR1_ON    0x08
#define CMD_BAR2_ON    0x09
#define CMD_BAR3_ON    0x0A
#define CMD_BAR4_ON    0x0B
#define CMD_BAR5_ON    0x0C
#define CMD_BAR6_ON    0x0D
#define CMD_BAR7_ON    0x0E
#define CMD_BAR8_ON    0x0F


/* Inputs to the battery control command */
/* There's an array in ic1.c (../../embedded/sbr5/ic1.c) */
/* which defines which command is in which position here */
#define CMD_BATT1_OFF     0x00
#define CMD_BATT2_OFF     0x01
#define CMD_BATT3_OFF     0x02
#define CMD_BATT4_OFF     0x03
#define CMD_BATT5_OFF     0x04
#define CMD_BATT6_OFF     0x05

#define CMD_BATT1_ON      0x06
#define CMD_BATT2_ON      0x07
#define CMD_BATT3_ON      0x08
#define CMD_BATT4_ON      0x09
#define CMD_BATT5_ON      0x0A
#define CMD_BATT6_ON      0x0B


/* Bits of the thruster state response */
#define THRUSTER1_ENABLED     0x01
#define THRUSTER2_ENABLED     0x02
#define THRUSTER3_ENABLED     0x04
#define THRUSTER4_ENABLED     0x08
#define THRUSTER5_ENABLED     0x10
#define THRUSTER6_ENABLED     0x20
#define ALL_THRUSTERS_ENABLED \
    (THRUSTER1_ENABLED | THRUSTER2_ENABLED | THRUSTER3_ENABLED | \
     THRUSTER4_ENABLED | THRUSTER5_ENABLED | THRUSTER6_ENABLED)

/* Overcurrent bits. Last 2 are marker droppers */
#define THRUSTER1_OVR     0x01
#define THRUSTER2_OVR     0x02
#define THRUSTER3_OVR     0x04
#define THRUSTER4_OVR     0x08
#define THRUSTER5_OVR     0x10
#define THRUSTER6_OVR     0x20


/* Bits of the bar state response */
#define BAR1_ENABLED    0x01
#define BAR2_ENABLED    0x02
#define BAR3_ENABLED    0x04
#define BAR4_ENABLED    0x08
#define BAR5_ENABLED    0x10
#define BAR6_ENABLED    0x20
#define BAR7_ENABLED    0x40
#define BAR8_ENABLED    0x80


/* Bits of the battery state response */
/* Ie, is the battery enabled? Does not imply the battery is actually in use */
/* For that, see below. */
#define BATT1_ENABLED      0x01
#define BATT2_ENABLED      0x02
#define BATT3_ENABLED      0x04
#define BATT4_ENABLED      0x08
#define BATT5_ENABLED      0x10
#define BATT6_ENABLED      0x20



/* Bits of the battery utilization response */
/* Ie, is the battery actually being used? */
#define BATT1_INUSE       0x01
#define BATT2_INUSE       0x02
#define BATT3_INUSE       0x04
#define BATT4_INUSE       0x08
#define BATT5_INUSE       0x10
#define BATT6_INUSE       0x20

#define ANIMATION_NONE      0
#define ANIMATION_REDBLUE   1
#define ANIMATION_REDGREEN  2

/* Servo defines */

/* Yes these are the same MotorBoard r3 has some wiring bugs */
#define SERVO_1      0
#define SERVO_2      1
#define SERVO_3      2
#define SERVO_4      3

#define SERVO_ENABLE_1        0x01
#define SERVO_ENABLE_2        0x02
#define SERVO_ENABLE_3_4      0x0c

#define SERVO_POWER_ON 1
#define SERVO_POWER_OFF 0

/* Bits of the status response */
/* Water is present */
#define STATUS_WATER      0x0080

/* Kill switch is attached */
#define STATUS_KILLSW     0x0100

/* Start switch is being pressed */
#define STATUS_STARTSW    0x0200

#define CMD_CAM_RELAY_ON 0x6C
#define CMD_CAM_RELAY_OFF 0x6D

namespace Tortuga
{
// If we are compiling as C++ code we need to use extern "C" linkage
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

 
enum partialUpdateType_
{
    NO_UPDATE,
    STATUS,
    THRUSTER_STATE,
    BAR_STATE,
    OVERCURRENT_STATE,
    BATTERY_ENABLES,
    TEMP,
    MOTOR_CURRENTS,
    BOARD_VOLTAGES_CURRENTS,
    BATTERY_VOLTAGES,
    BATTERY_CURRENTS,
    BATTERY_USED,
    SONAR,
    END_OF_UPDATES,
};

/** Information about the vehicles power state */
struct powerInfo
{
    /** Currents for motors and marker droppers */
    float motorCurrents[8];

    /** Voltage of 12V bus, in V. */
    float v12VBus;

    /** Voltage of 5V bus, in V */
    float v5VBus;

    /** Current of 12V bus, in A */
    float i12VBus;

    /** Current of 5V bus, in A */
    float i5VBus;

    /** Current of aux (carnetix) output, in A */
    float iAux;

    /** Voltage of balanced 26V, in V. NOT IMPLEMENTED IN BALANCER r2 */
    float v26VBus;

    /** 0-4 are batt 1-4. 5 is external power (batt 6). In V */
    float battVoltages[6];

    /** Battery currents. See note above. In A */
    float battCurrents[6];
};

struct sonarData
{
    /** These are good to 4 decimal places ONLY */
    double vectorX;
    /** These will always be in [-1, 1] */
    double vectorY;
    double vectorZ;
    unsigned char status;
    unsigned short range;
    /** Seconds part of the time when the ping occured */
    unsigned int timeStampSec;
    /** Microseconds part of the time when the ping occured */
    unsigned int timeStampUSec;
    /** Identifies which pinger the ping came from */
    unsigned char pingerID;
};

/** Complete vehicle information */
struct boardInfo
{
    /** What was last updates in this struct, value of partialUpdateType enum */
    enum partialUpdateType_ updateState;
    /** Status register- startsw, killsw, water, batt used */
    int status;
    /** Which thrusters are on */
    int thrusterState;
    /** Which bar outputs are on */
    int barState;
    /** Which thrusters have over-currented */
    int ovrState;
    /** Which batteries are enabled (not the same as in use) */
    int battEnabled;
    /** Which batteries are being drawn by the balancing circuit */
    int battUsed;

    /** Everything related to power. See above */
    struct powerInfo powerInfo;

    /** Temperatures, in deg C
     *
     * These are scattered throughout. The first one is the sensorboard temp.
     * The last two are distro and balancer temp (or vice versa?)
     * The middle ones are floaties, if we even have them connected
     */
    unsigned char temperature[NUM_TEMP_SENSORS];

    /** The latest data from the */
    struct sonarData sonar;
};


/* Perform next step of update cycle.
 * Returns: SB_OK on success
 *          SB_UPDATEDONE on success and update cycle is done
 *          SB_ERROR, SB_IOERROR, SB_BADCC, SB_HWFAIL, SB_ERROR on failure
 */
int partialRead(int fd, struct boardInfo * info);

/** Returns the file*/
int openSensorBoard(const char * devName);

/** Syncs the communication protocol between the board and vehicle */
int syncBoard(int fd);

int checkBoard(int fd);


int pingBoard(int fd);

/** Requests the depth value from the device (or error code)
 *
 *  @return An integer between 0 and 1023, or SB_ERROR.
 */
int readDepth(int fd);

/** Read the status bit back from the board */
int readStatus(int fd);

/** Reads the state of thrusters (safed or not)
 *  Returns a bit combination of THRUSTERx_ENABLED as above
 *  or SB_ERROR. How to tell them apart? SB_ERROR is negative,
 *  don't worry.
 */
int readThrusterState(int fd);

int hardKill(int fd);

/** This drops the marker (accepts only 0 and 1 for markerNum) */
int dropMarker(int fd, int markerNum);

/** Enables and disables the servo power supply
 *  @param fd
 *      The file descriptor returned by openSensorBoard()
 *  @param power
 *      A non-zero value turns on the servo power, and zero value turns off
 *      the servo power.
 */
int setServoPower(int fd, unsigned char power);

/** Sets the enable state of the each servo
 *
 *  @param fd
 *      The file descriptor returned by openSensorBoard()
 *  @param servoMask
 *      Sets the enable states of the servo, each bit of the byte that is 1
 *      turns activates a servo.  If the bit is 0 that servo signal is disabled.
 *      Currently we have only two servos, so bits 1 & 2 are the only usable
 *      ones.
 */    
int setServoEnable(int fd, unsigned char servoMask);

/** Sets the position of desired servo
 *
 *  @param fd
 *      The file descriptor returned by openSensorBoard()
 *  @param servoNumber
 *      The specific servo to turn on and off.
 *  @param position
 *      A 16 bit number that specifies the position of the servo. These numbers
 *      varry per servo so they must be specifically calibrated.
 *
 */
int setServoPosition(int fd, unsigned char servoNumber,
                     unsigned short position);


/** Power cycles the motor board */
int resetMotorBoard(int fd);
    
int lcdBacklight(int fd, int state);

/** Either enables or disables a desired thruster */
int setThrusterSafety(int fd, int state);

int setBarState(int fd, int state);

int displayText(int fd, int line, const char* text);

/**  Reads the values from the board's temperature

     @param fd
         The file descriptor returned by openSensorBoard()
     @param tempData
         Where the sensor data is written. The array must be at least
         NUM_TEMP_SENSORS elements long. The temperatures are specified in
         degrees C. A value of 255 indicates a missing or malfunctioning
         sensor.
     @return SB_OK upon success or SB_ERROR.
**/
int readTemp(int fd, unsigned char * tempData);

int getSonarData(int fd, struct sonarData * sd);


int startBlackfin(int fd);
int stopBlackfin(int fd);
int resetBlackfin(int fd);

int setDiagnostics(int fd, int state);

/** Set the speed of the thrusters

    This command takes about 2 ms to execute.  You must call
    readSpeedResponses before this command, or about 15 ms after this call is
    made.

    @param fd
         The file descriptor returned by openSensorBoard()

    @param s1
         The speed of thruster with address one
    @param s2
         The speed of thruster with address two
    @param s3
         The speed of thruster with address three
    @param s4
         The speed of thruster with address four
    @param s5
         The speed of thruster with address five
    @param s6
         The speed of thruster with address six
 */
int setSpeeds(int fd, int s1, int s2, int s3, int s4, int s5, int s6);

/** Reads back the on the board from the motor controller

    This is basically a house cleaning command, seee setSpeeds for information
    on its use.
 */
int readSpeedResponses(int fd);

int readThrusterState(int fd);

int readBarState(int fd);

int readOvrState(int fd);

int readBatteryEnables(int fd);

int readBatteryUsage(int fd);

int readMotorCurrents(int fd, struct powerInfo * info);
int readBoardVoltages(int fd, struct powerInfo * info);

int readBatteryVoltages(int fd, struct powerInfo * info);
int readBatteryCurrents(int fd, struct powerInfo * info);

int switchToExternalPower(int fd);
int switchToInternalPower(int fd);

int setBatteryState(int fd, int state);

int setAnimation(int fd, int anim);
int setBarOutputs(int fd, int bars);

// maxCurrent (mA) = (a * speed) / 6 + b*40
// where speed=[0,255] corresponds to 0 to full speed
int setOvrParams(int fd, int a, int b);
int readOvrParams(int fd, int * a, int * b);

/** Translates the function error return codes into text */
char* sbErrorToText(int ret);

/** Translates the index from the boardInfo array into the sensor name */
char* tempSensorIDToText(int id);

int DVLOn(int fd, unsigned char power);

int setMagPower(int fd, unsigned char power);
int fireTorpedo(int fd, unsigned char torpnum);
int voidTorpedo(int fd, unsigned char torpnum);
int armTorpedo(int fd, unsigned char torpnum);
int extendGrabber(int fd, int param);
int retractGrabber(int fd);
int voidGrabber(int fd);
int voidSystem(int fd);
int pneumaticsOff(int fd);
int setDerpyPower(int fd, unsigned char on);
int setDerpySpeed(int fd, int speed);
int stopDerpy(int fd);
int camConnect(int fd);
int camDisconnect(int fd);


 /*  @return An integer between 0 and 1023, or SB_ERROR.
 */
int readDepth(int fd);

/** Read the status bit back from the board */
int readStatus(int fd);

/** Reads the state of thrusters (safed or not)
 *  Returns a bit combination of THRUSTERx_ENABLED as above
 *  or SB_ERROR. How to tell them apart? SB_ERROR is negative,
 *  don't worry.
 */
int readThrusterState(int fd);

int hardKill(int fd);

/** This drops the marker (accepts only 0 and 1 for markerNum) */
int dropMarker(int fd, int markerNum);

/** Enables and disables the servo power supply
 *  @param fd
 *      The file descriptor returned by openSensorBoard()
 *  @param power
 *      A non-zero value turns on the servo power, and zero value turns off
 *      the servo power.
 */
int setServoPower(int fd, unsigned char power);

/** Sets the enable state of the each servo
 *
 *  @param fd
 *      The file descriptor returned by openSensorBoard()
 *  @param servoMask
 *      Sets the enable states of the servo, each bit of the byte that is 1
 *      turns activates a servo.  If the bit is 0 that servo signal is disabled.
 *      Currently we have only two servos, so bits 1 & 2 are the only usable
 *      ones.
 */    
int setServoEnable(int fd, unsigned char servoMask);

/** Sets the position of desired servo
 *
 *  @param fd
 *      The file descriptor returned by openSensorBoard()
 *  @param servoNumber
 *      The specific servo to turn on and off.
 *  @param position
 *      A 16 bit number that specifies the position of the servo. These numbers
 *      varry per servo so they must be specifically calibrated.
 *
 */
int setServoPosition(int fd, unsigned char servoNumber,
                     unsigned short position);


/** Power cycles the motor board */
int resetMotorBoard(int fd);
    
int lcdBacklight(int fd, int state);

/** Either enables or disables a desired thruster */
int setThrusterSafety(int fd, int state);

int setBarState(int fd, int state);

int displayText(int fd, int line, const char* text);

/**  Reads the values from the board's temperature

     @param fd
         The file descriptor returned by openSensorBoard()
     @param tempData
         Where the sensor data is written. The array must be at least
         NUM_TEMP_SENSORS elements long. The temperatures are specified in
         degrees C. A value of 255 indicates a missing or malfunctioning
         sensor.
     @return SB_OK upon success or SB_ERROR.
**/
int readTemp(int fd, unsigned char * tempData);

int getSonarData(int fd, struct sonarData * sd);


int startBlackfin(int fd);
int stopBlackfin(int fd);
int resetBlackfin(int fd);

int setDiagnostics(int fd, int state);

/** Set the speed of the thrusters

    This command takes about 2 ms to execute.  You must call
    readSpeedResponses before this command, or about 15 ms after this call is
    made.

    @param fd
         The file descriptor returned by openSensorBoard()

    @param s1
         The speed of thruster with address one
    @param s2
         The speed of thruster with address two
    @param s3
         The speed of thruster with address three
    @param s4
         The speed of thruster with address four
    @param s5
         The speed of thruster with address five
    @param s6
         The speed of thruster with address six
 */
int setSpeeds(int fd, int s1, int s2, int s3, int s4, int s5, int s6);

/** Reads back the on the board from the motor controller

    This is basically a house cleaning command, seee setSpeeds for information
    on its use.
 */
int readSpeedResponses(int fd);

int readThrusterState(int fd);

int readBarState(int fd);

int readOvrState(int fd);

int readBatteryEnables(int fd);

int readBatteryUsage(int fd);

int readMotorCurrents(int fd, struct powerInfo * info);
int readBoardVoltages(int fd, struct powerInfo * info);

int readBatteryVoltages(int fd, struct powerInfo * info);
int readBatteryCurrents(int fd, struct powerInfo * info);

int switchToExternalPower(int fd);
int switchToInternalPower(int fd);

int setBatteryState(int fd, int state);

int setAnimation(int fd, int anim);
int setBarOutputs(int fd, int bars);

// maxCurrent (mA) = (a * speed) / 6 + b*40
// where speed=[0,255] corresponds to 0 to full speed
int setOvrParams(int fd, int a, int b);
int readOvrParams(int fd, int * a, int * b);

/** Translates the function error return codes into text */
char* sbErrorToText(int ret);

/** Translates the index from the boardInfo array into the sensor name */
char* tempSensorIDToText(int id);

int DVLOn(int fd, unsigned char power);

int setMagPower(int fd, unsigned char power);
int fireTorpedo(int fd, unsigned char torpnum);
int voidTorpedo(int fd, unsigned char torpnum);
int armTorpedo(int fd, unsigned char torpnum);
int extendGrabber(int fd, int param); //kanga
int retractGrabber(int fd);
int voidGrabber(int fd);
int voidSystem(int fd);
int pneumaticsOff(int fd);
int setDerpyPower(int fd, unsigned char on);
int setDerpySpeed(int fd, int speed);
int stopDerpy(int fd);

/*Turn the camera connection on or off
kanga - 7/3/2013*/
int camConnect(int fd);
int camDisconnect(int fd);

    }

// If we are compiling as C++ code we need to use extern "C" linkage
#ifdef __cplusplus
} extern "C"
#endif // __cplusplus

#endif // RAM_DRIVER_SENSORAPI_H_06_09_2008
