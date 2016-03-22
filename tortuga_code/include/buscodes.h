/*
 * Commands Master can send to Slaves. Command numbers among different Slaves can overlap
 * but common commands like Ping/Identify should be the same everywhere.
 * Kind of obvious what order I did these in, huh?
 */
#define BUS_CMD_PING            0x00
#define BUS_CMD_ID              0x01
#define BUS_CMD_READ_REG        0x02
#define BUS_CMD_WRITE_REG       0x03
#define BUS_CMD_MARKER1         0x04
#define BUS_CMD_DEPTH           0x05
#define BUS_CMD_LCD_WRITE       0x06
#define BUS_CMD_LCD_REFRESH     0x07
#define BUS_CMD_LCD_LIGHT_ON    0x08
#define BUS_CMD_LCD_LIGHT_OFF   0x09
#define BUS_CMD_THRUSTERS_ON    0x0A
#define BUS_CMD_THRUSTERS_OFF   0x0B
#define BUS_CMD_MARKER2         0x0C
#define BUS_CMD_CHECKWATER      0x0E
#define BUS_CMD_TEMP            0x0F
#define BUS_CMD_BOARDSTATUS     0x10
#define BUS_CMD_HARDKILL        0x11
#define BUS_CMD_LCD_LIGHT_FLASH 0x12

#define BUS_CMD_THRUSTER1_OFF   0x13
#define BUS_CMD_THRUSTER2_OFF   0x14
#define BUS_CMD_THRUSTER3_OFF   0x15
#define BUS_CMD_THRUSTER4_OFF   0x16
#define BUS_CMD_THRUSTER5_OFF   0x23
#define BUS_CMD_THRUSTER6_OFF   0x24


#define BUS_CMD_THRUSTER1_ON    0x17
#define BUS_CMD_THRUSTER2_ON    0x18
#define BUS_CMD_THRUSTER3_ON    0x19
#define BUS_CMD_THRUSTER4_ON    0x1A
#define BUS_CMD_THRUSTER5_ON    0x25
#define BUS_CMD_THRUSTER6_ON    0x26


#define BUS_CMD_THRUSTER_STATE  0x1C

#define BUS_CMD_SONAR           0x1B

#define BUS_CMD_CLEARU1RX       0x1D
#define BUS_CMD_CLEARU2RX       0x1E

/* Followed by 2 parameters. MSB, LSB
 * Command goes out when LSB comes in.
 */
#define BUS_CMD_SETSPEED_U1     0x1F
#define BUS_CMD_SETSPEED_U2     0x20

/* Returns 1 byte length followed by 1st byte of response
 * TODO: This should be better, but right now, motor controllers
 * only reply with 06 or nothing
 */
#define BUS_CMD_GETREPLY_U1     0x21
#define BUS_CMD_GETREPLY_U2     0x22

#define BUS_CMD_STARTSW         0x27


/* Battery enables on the balancer board */
#define BUS_CMD_BATT1_OFF       0x28
#define BUS_CMD_BATT2_OFF       0x29
#define BUS_CMD_BATT3_OFF       0x2A
#define BUS_CMD_BATT4_OFF       0x2B
#define BUS_CMD_BATT5_OFF       0x2C
#define BUS_CMD_BATT6_OFF       0x51

#define BUS_CMD_BATT1_ON        0x2D
#define BUS_CMD_BATT2_ON        0x2E
#define BUS_CMD_BATT3_ON        0x2F
#define BUS_CMD_BATT4_ON        0x30
#define BUS_CMD_BATT5_ON        0x31
#define BUS_CMD_BATT6_ON        0x52


/* Bars (multi-color LEDs) */
#define BUS_CMD_BAR_COLOR       0x5A

#define BUS_CMD_BAR1_OFF        0x32
#define BUS_CMD_BAR2_OFF        0x33
#define BUS_CMD_BAR3_OFF        0x34
#define BUS_CMD_BAR4_OFF        0x35
#define BUS_CMD_BAR5_OFF        0x36
#define BUS_CMD_BAR6_OFF        0x37
#define BUS_CMD_BAR7_OFF        0x38
#define BUS_CMD_BAR8_OFF        0x39

#define BUS_CMD_BAR1_ON         0x3A
#define BUS_CMD_BAR2_ON         0x3B
#define BUS_CMD_BAR3_ON         0x3C
#define BUS_CMD_BAR4_ON         0x3D
#define BUS_CMD_BAR5_ON         0x3E
#define BUS_CMD_BAR6_ON         0x3F
#define BUS_CMD_BAR7_ON         0x40
#define BUS_CMD_BAR8_ON         0x41

#define BUS_CMD_BAR_STATE       0x42
#define BUS_CMD_READ_OVR        0x43

#define BUS_CMD_READ_IMOTOR     0x44

#define BUS_CMD_READ_ASTATUS    0x45

#define BUS_CMD_BATTSTATE       0x46

#define BUS_CMD_BATTVOLTAGE     0x47

#define BUS_CMD_BATTCURRENT     0x48

#define BUS_CMD_EXTPOWER        0x49
#define BUS_CMD_INTPOWER        0x4A

#define BUS_CMD_MOTRSPEEDS      0x4B

//#define BUS_CMD_SET_BARMODE     0x4C
#define BUS_CMD_BFRESET         0x4D

#define BUS_CMD_SET_BARS        0x4E

#define BUS_CMD_BFIN_STOP       0x4F
#define BUS_CMD_BFIN_START      0x50

/* NOTE: Since there was no space for BATT6_ON and
 * BATT6_OFF I used 0x51 and 0x52 above
 * --Kit (5/23/2009) */

#define BUS_CMD_SET_MOT_SPEEDS  0x53
#define BUS_CMD_KILL_MOTORS     0x54
#define BUS_CMD_SET_MOT_N       0x55

/* Servo power is now depreciated, but to avoid breaking
   working code I'm going to keep them around.  "Magnet"
   power is what is replacing it, which is for the magnetic
   droppers.

   Magnetic droppers have been replaced by pneumatic droppers.
   0x56 and 0x57 should now be depreciated
 */
#define BUS_CMD_SERVO_POWER_ON  0x56
#define BUS_CMD_MAG_PWR_ON      0x56
#define BUS_CMD_SERVO_POWER_OFF 0x57
#define BUS_CMD_MAG_PWR_OFF     0x57

/* These are both depreciated */
#define BUS_CMD_SERVO_ENABLE    0x58
#define BUS_CMD_SET_SERVO_POS   0x59

/* 0x5A is taken by BUS_CMD_BAR COLOR above */

#define BUS_CMD_DVL_ON          0x5B
#define BUS_CMD_DVL_OFF         0x5C

#define BUS_CMD_FIRE_TORP_1     0x5D
#define BUS_CMD_FIRE_TORP_2     0x5E
#define BUS_CMD_VOID_TORP_1     0x5F
#define BUS_CMD_VOID_TORP_2     0x60
#define BUS_CMD_ARM_TORP_1      0x61
#define BUS_CMD_ARM_TORP_2      0x62

#define BUS_CMD_EXT_GRABBER     0x63
#define BUS_CMD_RET_GRABBER     0x64
#define BUS_CMD_VOID_GRABBER    0x65

#define BUS_CMD_VOID_PNEU       0x66
#define BUS_CMD_OFF_PNEU        0x67

/* Derpy commands! */
/* Stop derpy is automatically generated when you send a DERPY_OFF cmd */
#define BUS_CMD_DERPY_ON        0x68
#define BUS_CMD_DERPY_OFF       0x69
#define BUS_CMD_SET_DERPY       0x6A
#define BUS_CMD_STOP_DERPY      0x6B


/* In order to fix the in-rush issue with the power supply inside the cameras 
we are setting up an interrupt circuit for replicating plugging and unplugging 
the subcon connector. this will be fixed when the power supply for the cameras 
is redesigned

Kanga - 7/3/2013*/
#define BUS_CMD_CAM_RELAY_ON    0x6C
#define BUS_CMD_CAM_RELAY_OFF   0x6D

//Kanga - Enabling options for separately extending grabbers

 #define BUS_CMD_EXT_GRABBER_1 	0x6E
 #define BUS_CMD_EXT_GRABBER_2	0x6F

/*Next free is 0x70*/

/* I wanted a more Unique response to a ping.
 * Ideally, the response would be 0xBEEF or 0xDEAD or 0xABADBABE
 * Perhaps 0xFEEF1F0 like FEE F1 F0 FUM or 0xAC like the start of ACK
 *
 * For now, I think the response will have to be 0xA5 which is 10100101
 * which, while not as fun, is symmetric, and hard to have happen randomly. */
#define BUS_CMD_PING_RESP 0xA5;


/* Host commands are commands which the computron sends to senior ic1.
 * --Kit (5/23/2009) */
#define HOST_CMD_SYNC               0xFF

#define HOST_CMD_PING               0x00
#define HOST_REPLY_SUCCESS          0xBC
#define HOST_REPLY_FAILURE          0xDF
#define HOST_REPLY_BADCHKSUM        0xCC

#define HOST_CMD_SYSCHECK           0x01

#define HOST_CMD_DEPTH              0x02
#define HOST_REPLY_DEPTH            0x03

#define HOST_CMD_BOARDSTATUS        0x04
#define HOST_REPLY_BOARDSTATUS      0x05

#define HOST_CMD_HARDKILL           0x06
#define HOST_CMD_MARKER             0x07

#define HOST_CMD_BACKLIGHT          0x08

#define HOST_CMD_THRUSTERS          0x09

#define HOST_CMD_TEMPERATURE        0x0A
#define HOST_REPLY_TEMPERATURE      0x0B

#define HOST_CMD_PRINTTEXT          0x0C

#define HOST_CMD_SONAR              0x0D
#define HOST_REPLY_SONAR            0x0E

#define HOST_CMD_RUNTIMEDIAG        0x0F

#define HOST_CMD_THRUSTERSTATE      0x10
#define HOST_REPLY_THRUSTERSTATE    0x11

#define HOST_CMD_SETSPEED           0x12

#define HOST_CMD_MOTOR_READ         0x13
#define HOST_CMD_MOTOR_REPLY        0x14


#define HOST_CMD_BARS               0x15

#define HOST_CMD_BARSTATE           0x16
#define HOST_REPLY_BARSTATE         0x17

#define HOST_CMD_IMOTOR             0x18
#define HOST_REPLY_IMOTOR           0x19

#define HOST_CMD_VLOW               0x1A
#define HOST_REPLY_VLOW             0x1B

#define HOST_CMD_READ_OVR           0x1C
#define HOST_REPLY_OVR              0x1D

#define HOST_CMD_BATTSTATE          0x1E
#define HOST_REPLY_BATTSTATE        0x1F

#define HOST_CMD_BATTCTL            0x20

#define HOST_CMD_BATTVOLTAGE        0x21
#define HOST_REPLY_BATTVOLTAGE      0x22

#define HOST_CMD_BATTCURRENT        0x23
#define HOST_REPLY_BATTCURRENT      0x24

#define HOST_CMD_SWITCHPOWER        0x25

#define HOST_CMD_READ_OVRLIMIT      0x26
#define HOST_REPLY_OVRLIMIT         0x27

#define HOST_CMD_SET_OVRLIMIT       0x28

#define HOST_CMD_BFRESET            0x29

#define HOST_CMD_BARANIMATION       0x2A

#define HOST_CMD_SET_BARS           0x2B

#define HOST_CMD_BFIN_STATE         0x2C

/* These two servo commands are depreciated */
#define HOST_CMD_SERVO_ENABLE       0x2D
#define HOST_CMD_SET_SERVO_POS      0x2E

/* Servo power is gone, replaced by magnetic dropper
   power */
#define HOST_CMD_SERVO_POWER_ON     0x2F
#define HOST_CMD_MAG_PWR_ON         0x2F
#define HOST_CMD_SERVO_POWER_OFF    0x30
#define HOST_CMD_MAG_PWR_OFF        0x30

#define HOST_CMD_DVL_ON             0x32
#define HOST_CMD_DVL_OFF            0x33

#define HOST_CMD_FIRE_TORP_1        0x34
#define HOST_CMD_FIRE_TORP_2        0x35
#define HOST_CMD_VOID_TORP_1        0x36
#define HOST_CMD_VOID_TORP_2        0x37
#define HOST_CMD_ARM_TORP_1         0x38
#define HOST_CMD_ARM_TORP_2         0x39

#define HOST_CMD_EXT_GRABBER        0x3A
#define HOST_CMD_RET_GRABBER        0x3B
#define HOST_CMD_VOID_GRABBER       0x3C

#define HOST_CMD_VOID_PNEU          0x3D
#define HOST_CMD_OFF_PNEU           0x3E

#define HOST_CMD_DERPY_ON           0x3F
#define HOST_CMD_DERPY_OFF          0x40
#define HOST_CMD_SET_DERPY          0x41
#define HOST_CMD_STOP_DERPY         0x42


/*host commands for switching the camera connection on and off via relay
kanga - 7/3/2013*/

#define HOST_CMD_CAM_RELAY_ON    0x43
#define HOST_CMD_CAM_RELAY_OFF   0x44

//Kanga - enabling options for separately extending grabbers

 #define HOST_CMD_EXT_GRABBER_1 	0x45
 #define HOST_CMD_EXT_GRABBER_2		0x46

/* So we have host commands, Bus commands,
 * and then we had a section with commands.
 *
 * These turned out to be a reference to the array
 * for turning batteries and bars on and off for
 * lcdshow.  They're defined in
 * ../../drivers/sensor-r5/include/sensorapi.h
 *
 * --Kit (5/23/2009) */


/* Another mysterious define, Hooray! This one *sounds*
 * self-explanitory, and yet, is it important? Why is
 * it in Buscodes.h?
 * --Kit (5/23/2009) */
#define SONAR_PACKET_LEN 22
