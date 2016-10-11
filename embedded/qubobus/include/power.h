#include <stdint.h>

#ifndef QUBOBUS_POWER_H
#define QUBOBUS_POWER_H

/* Host -> QSCU requesting a power status message. */
#define D_ID_POWER_STATUS_REQUEST 40

/* QSCU -> Host reply containing a power status struct. */
#define D_ID_POWER_STATUS 41
struct Power_Status {
    /* Measurements for the 3.3-volt common rail. */
    float rail_3_V;
    float rail_3_A;

    /* Measurements for the 5-volt common rail. */
    float rail_5_V;
    float rail_5_A;

    /* Measurements for the 9-volt common rail. */
    float rail_9_V;
    float rail_9_A;

    /* Measurements for the 12-volt common rail. */
    float rail_12_V;
    float rail_12_A;

    /* Measurements for the 16-volt common rail. */
    float rail_16_V;
    float rail_16_A;

    /* Measurements for the first battery input. */
    float batt_0_V;
    float batt_0_A;
    
    /* Measurements for the second battery input. */
    float batt_1_V;
    float batt_1_A;

    /* Measurements for the shore power input. */
    float shore_V;
    float shore_A;
};

/* Host -> QSCU set the software limits for determining when to send error messages. */
#define D_ID_POWER_CONFIG 42
struct Power_Config {
    /* Limits for the 3.3-volt common rail. */
    float rail_3_low_V;
    float rail_3_high_V;
    float rail_3_high_A;

    /* Limits for the 5-volt common rail. */
    float rail_5_low_V;
    float rail_5_high_V;
    float rail_5_high_A;

    /* Limits for the 9-volt common rail. */
    float rail_9_low_V;
    float rail_9_high_V;
    float rail_9_high_A;

    /* Limits for the 12-volt common rail. */
    float rail_12_low_V;
    float rail_12_high_V;
    float rail_12_high_A;

    /* Limits for the 16-volt common rail. */
    float rail_16_low_V;
    float rail_16_high_V;
    float rail_16_high_A;

    /* Limits for the first battery input. */
    float batt_0_low_V;
    float batt_0_high_V;
    float batt_0_high_A;

    /* Limits for the second battery input. */
    float batt_1_low_V;
    float batt_1_high_V;
    float batt_1_high_A;

    /* Limits for the shore power input. */
    float shore_low_V;
    float shore_high_V;
    float shore_high_A;
};

/* Defintions for different rail IDs. */
#define RAIL_ID_3V 0
#define RAIL_ID_5V 1
#define RAIL_ID_9V 2
#define RAIL_ID_12V 3
#define RAIL_ID_DVL 4

/* Host -> QSCU command to enable the power supply for a specified rail */
#define D_ID_POWER_ENABLE 43
struct Power_Enable {
    uint8_t rail_id;   
};

/* Host -> QSCU command to disable the power supply for a specified rail. */
#define D_ID_POWER_DISABLE 44
struct Power_Disable {
    uint8_t rail_id;
};

/* Error sent when the power board is unreachable. */
#define E_ID_POWER_UNREACHABLE 40

/* Error sent when the power board notices an overvoltage condition. */
#define E_ID_POWER_OVERVOLT 41

/* Error sent when the powre board notices an undervoltage condition. */
#define E_ID_POWER_UNDERVOLT 42

/* Error sent when the power board notices excessive current draw. */
#define E_ID_POWER_OVERCURRENT 43

#endif
