
/**
 * All of the following types will be assembled as-is, with no padding. 
 * The paired statement is pack(pop)
 */
#pragma pack(push,1)

public:
/************************************************************************
 * PRIMITIVE TYPEDEFS
 * Basic primitive types given more specific names.
 * Reflects the different data type sizes in low-level protocols.
 ************************************************************************/
/** Identifier for commands, sent at the beginning of a frame. */
typedef uint8_t frameid_t;
/** Length of data sent in a single packet. */
typedef uint16_t bytecount_t;
/** Xmodem 16-bit CRC checksum sent at the end of a packet. */
typedef uint16_t checksum_t;

private:
/************************************************************************
 * INTERNAL STRUCT TYPEDEFS
 * Structs used for internal organization and communication.
 ************************************************************************/

/** Reference details about a command sent over serial. */
typedef struct _Command {
   /** printf format string for the command itself. */
   const char* format;
   /** Name of the command for debugging. */
   const char* name;
} Command;

typedef std::vector<char> Payload;

/** Message read/written from the hardware device. */
typedef struct _Message {
   /** Frame identifier from the beginning of the data frame. */
   frameid_t id;
   /** Length of the payload pointed to by the payload pointer. */
   bytecount_t payload_size;
   /** Pointer to the memory allocated for this message. */
   std::shared_ptr<Payload> payload;
} Message;

/******************************************************************************
 * HARDCODED PAYLOAD TYPES:
 * These should be exactly out of the specification file.
 ******************************************************************************/

/** Header for each packet, containing information about the whole packet. */
typedef struct _PD0Header {
   uint8_t header_id;
   uint8_t data_source_id;
   bytecount_t bytes_in_ensemble;
   uint8_t spare;
   uint8_t data_types;
} PD0Header;

typedef uint16_t data_offset_t;

/** Fixed leader data containing static config details */
typedef struct _PD0FixedLeaderData {
   uint16_t leader_id;
   uint8_t cpu_fw_ver;
   uint8_t cpu_fw_rev;
   uint16_t system_config;
   uint8_t simulation_flag;
   uint8_t lag_length;
   uint8_t nr_beams;
   uint8_t nr_cells;
   uint16_t pings_per_ensemble;
   uint16_t depth_cell_length;
   uint16_t blank_after_transmit;
   uint8_t profiling_mode;
   uint8_t low_corr_threshold;
   uint8_t nr_code_reps;
   uint8_t pct_good_minimum;
   uint16_t err_vel_max;
   uint8_t tpp_minutes;
   uint8_t tpp_seconds;
   uint8_t tpp_hundreths;
   uint8_t coord_transform;
   uint16_t heading_alignment;
   uint16_t heading_bias;
   uint8_t sensor_source;
   uint16_t bin1_distance;
   uint16_t xmit_pulse_length;
   uint16_t wp_ref_layer_avg;
   uint8_t false_target_threshold;
   uint8_t spare0;
   uint16_t xmit_lag_distance;
   uint8_t spare1[8];
   uint16_t system_bandwith;
   uint16_t spare2;
   uint32_t serial_number;
} PD0FixedLeaderData;

/** Variable leader data containing information about the current reading. */
typedef struct _PD0VariableLeaderData {
   uint16_t leader_id;
   uint16_t ensemble_number;
   uint8_t rtc_year;
   uint8_t rtc_month;
   uint8_t rtc_day;
   uint8_t rtc_hour;
   uint8_t rtc_minute;
   uint8_t rtc_second;
   uint8_t rtc_hundreths;
   uint8_t ensemble_number_msb;
   uint16_t bit_result;
   uint16_t speed_of_sound;
   uint16_t depth;
   uint16_t heading;
   uint16_t pitch;
   uint16_t roll;
   uint16_t salinity;
   uint16_t temperature;
   uint8_t mpt_minutes;
   uint8_t mpt_seconds;
   uint8_t mpt_hundreths;
   uint8_t heading_stddev;
   uint8_t pitch_stddev;
   uint8_t roll_stddev;
   uint8_t adc_channel[8];
   uint32_t error_status;
   uint16_t spare0;
   uint32_t pressure;
   uint32_t pressure_variance;
   uint32_t spare1;
} PD0VariableLeaderData;

/** Number of depth cells to compensate for */
#define NUM_CELLS 128

/** Pack of velocity data for each depth cell */
typedef short int[4] cell_velocity_t;

/** Velocity data format */
typedef struct _VelocityData {
   uint16_t velocity_id;
   cell_velocity_t cells[NUM_CELLS];
} VelocityData;

/** Pack of status data for each cell in a reading */
typedef uint8_t[4] data_status_t;

/** Status data format, has information about each depth cell's readings. */
typedef struct _DataStatus {
   uint16_t status_id;
   data_status_t cells[NUM_CELLS];
} DataStatus;

/** Bottom tracking data */
typedef struct _PD0BottomTrackData {
   uint16_t bottom_track_id;
   uint16_t pings_per_ensemble;
   uint16_t reserved0;
   uint8_t corr_mag_min;
   uint8_t eval_amp_min;
   uint8_t reserved1;
   uint8_t mode;
   uint16_t velocity_max;
   uint32_t reserved2;
   uint16_t bot_range[4];
   uint16_t bot_velocity[4];
   uint8_t bot_correlation[4];
   uint8_t bot_eval_amp[4];
   uint8_t bot_pct_good[4];
   uint16_t ref_min;
   uint16_t ref_near;
   uint16_t ref_far;
   uint16_t ref_velocity[4];
   uint8_t ref_correlation[4];
   uint8_t ref_int[4];
   uint8_t ref_pct_good[4];
   uint16_t bot_max_depth;
   uint8_t rssi_amp[4];
   uint8_t gain;
   uint8_t range_msb[4];
} PD0BottomTrackData;

/** Environmental parameters */
typedef struct _PD0EnvironmentData {
   uint16_t attitude_id;
   uint8_t attitude_output[8];
   uint8_t reserved;
   uint16_t fixed_heading;
   uint8_t fixed_heading_coord_frame;
   uint16_t roll_offset;
   uint16_t pitch_offset;
   uint8_t pitch_roll_frame[5];
   uint8_t orientation;
   uint16_t heading_bias;
   uint8_t sensor_source[8];
   uint32_t depth;
   uint8_t salinity;
   short int temperature;
   uint16_t speed_of_sound;
   uint8_t coordinate_transformation;
   uint8_t three_beam_solution;
   uint8_t bin_map;
   uint8_t msb_coordinate_transformation;
} PD0EnvironmentData;

/** Binary Data Format PD4 */
typedef struct _PD4Data {
   uint8_t data_id;
   uint8_t data_structure;
   uint16_t bytecount;
   uint8_t system_config;
   uint16_t x_vel;
   uint16_t y_vel;
   uint16_t z_vel;
   uint16_t e_vel;
   uint16_t beam_range[4];
   uint8_t bottom_status;
   uint16_t x_ref_vel;
   uint16_t y_ref_vel;
   uint16_t z_ref_vel;
   uint16_t e_ref_vel;
   uint16_t ref_start;
   uint16_t ref_end;
   uint8_t ref_status;
   uint8_t tofp_hour;
   uint8_t tofp_minute;
   uint8_t tofp_second;
   uint8_t tofp_hundreths;
   uint16_t bit_results;
   uint16_t speed_of_sound;
   uint16_t temperature;
   checksum_t checksum;
} PD4Data;

/** Binary Data Format PD5 */
typedef struct _PD5Data {
   uint8_t data_id;
   uint8_t data_structure;
   uint16_t bytecount;
   uint8_t system_config;
   uint16_t x_vel;
   uint16_t y_vel;
   uint16_t z_vel;
   uint16_t e_vel;
   uint16_t beam_range[4];
   uint8_t bottom_status;
   uint16_t x_ref_vel;
   uint16_t y_ref_vel;
   uint16_t z_ref_vel;
   uint16_t e_ref_vel;
   uint16_t ref_start;
   uint16_t ref_end;
   uint8_t ref_status;
   uint8_t tofp_hour;
   uint8_t tofp_minute;
   uint8_t tofp_second;
   uint8_t tofp_hundreths;
   uint16_t bit_results;
   uint16_t speed_of_sound;
   uint16_t temperature;
   uint8_t salinity;
   uint16_t depth;
   uint16_t pitch;
   uint16_t roll;
   uint16_t heading;
   uint32_t btm_distance_east;
   uint32_t btm_distance_north;
   uint32_t btm_distance_up;
   uint32_t btm_distance_error;
   uint32_t ref_distance_east;
   uint32_t ref_distance_north;
   uint32_t ref_distance_up;
   uint32_t ref_distance_error;
   checksum_t checksum;
} PD5Data;

public:
/************************************************************************
 * ENUMERATED TYPEDEFS
 * Enumerated types defining values for fields that can only take a
 * limited number of unique values.
 ************************************************************************/


/************************************************************************
 * API TYPEDEFS
 * Structs used to communicate with the user different information.
 ************************************************************************/

/** Specification for the baudrate of the serial link. */
typedef struct _DVLSpeed {
   /** Identifier for the device speed for the device. */
   uint8_t id;
   /** Identifier for the host speed for the computer. */
   speed_t baud;
} DVLSpeed;

/******************************************************************************
 * USER DEFINED TYPEDEFS
 * These types can be manipulated as to change the behavior of data flow.
 * Integrity and book-keeping are paramount, as many data payloads need a certain
 * format in order to be interpreted correctly.
 ******************************************************************************/


#pragma pack(pop)
