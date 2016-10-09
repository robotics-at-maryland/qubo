#include <stdint.h>

#ifndef QUBOBUS_POWER_H
#define QUBOBUS_POWER_H

#define D_ID_POWER_STATUS_REQUEST 40

#define D_ID_POWER_STATUS 41
struct Power_Status {
    /* Measurements for the 3.3-volt common rail. */
    float rail_3_V;
    float rail_3_mA;

    /* Measurements for the 5-volt common rail. */
    float rail_5_V;
    float rail_5_mA;

    /* Measurements for the 9-volt common rail. */
    float rail_9_V;
    float rail_9_mA;

    /* Measurements for the 12-volt common rail. */
    float rail_12_V;
    float rail_12_mA;

    /* Measurements for the 16-volt common rail. */
    float rail_16_V;
    float rail_16_mA;

    /* Measurements for the first battery input. */
    float batt_0_V;
    float batt_0_mA;
    
    /* Measurements for the second battery input. */
    float batt_1_V;
    float batt_1_mA;

    /* Measurements for the shore power input. */
    float shore_V;
    float shore_mA;
};

#define RAIL_ID_3V 0
#define RAIL_ID_5V 1
#define RAIL_ID_9V 2
#define RAIL_ID_12V 3
#define RAIL_ID_DVL 4

#define D_ID_POWER_ENABLE 42
struct Power_Enable {
    uint8_t rail_id;   
};

#define D_ID_POWER_DISABLE 43
struct Power_Disable {
    uint8_t rail_id;
};

#define ERR_ID_POWER_UNREACHABLE 44
#endif
