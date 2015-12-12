
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
typedef uint16_t frameid_t;
/** Length of data sent in a single packet. */
typedef uint16_t bytecount_t;
/** Xmodem 16-bit CRC checksum sent at the end of a packet. */
typedef uint16_t checksum_t;

private:
/******************************************************************************
 * HARDCODED PAYLOAD TYPES:
 * These should be exactly out of the specification file.
 ******************************************************************************/

/** Header for each packet, containing information about the whole packet. */
typedef struct _PD0_Header {
   bytecount_t bytes_in_ensemble;
   uint8_t spare;
   uint8_t data_types;
} PD0_Header;

typedef uint16_t data_offset_t;

/** Fixed leader data containing static config details */
typedef struct _PD0_FixedLeader {
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
} PD0_FixedLeader;

/** Variable leader data containing information about the current reading. */
typedef struct _PD0_VariableLeader {
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
} PD0_VariableLeader;

typedef short int PD0_CellShortFields[4];

typedef uint8_t PD0_CellByteFields[4];

/** Bottom tracking data */
typedef struct _PD0_BottomTrack {
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
} PD0_BottomTrack;

/** Environmental parameters */
typedef struct _PD0_Environment {
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
} PD0_Environment;

typedef struct _PD0_BottomTrackCommand {
   uint8_t tofp_hour;
   uint8_t tofp_minute;
   uint8_t tofp_second;
   uint8_t tofp_hundreth;
   uint8_t eval_amplitude;
   uint8_t correlation;
   uint16_t delay_before_reacq;
   uint16_t error_vel_max;
   uint16_t depth_guess;
   uint8_t pct_good_min;
   uint8_t gain_threshold_low;
   uint8_t gain_threshold_high;
   uint16_t gain_switch_depth;
   uint8_t water_mass_mode;
   uint16_t water_mass_layer_size;
   uint16_t water_mass_near;
   uint16_t water_mass_far;
   uint8_t bottom_track_mode;
   uint8_t speed_log_param_1;
   uint16_t speed_log_timeout;
   uint8_t speed_log_filter_time_constant;
   uint16_t pings_per_ensemble;
   uint8_t vertical_depth_resolution;
   uint16_t terrain_bias_correction;
   uint16_t bottom_blank;
   uint8_t correlation_threshold;
   uint8_t short_lag_control;
   uint16_t max_tracking_depth;
   uint16_t water_reference_interval;
   uint8_t max_transmit_percent;
   uint16_t ambiguity_velocity;
} PD0_BottomTrackCommand;

typedef struct _PD0_BottomTrackHighRes {
   uint32_t bot_velocity[4];
   uint32_t bot_distance[4];
   uint32_t ref_velocity[4];
   uint32_t ref_distance[4];
   uint32_t speed_of_sound;
} PD0_BottomTrackHighRes;

typedef struct _PD0_BottomTrackRange {
   uint32_t slant_range;
   uint32_t delta_range;
   uint32_t vertical_range;
   uint8_t percent_good_slant;
   uint8_t percent_good_onetwo;
   uint8_t percent_good_threefour;
   uint32_t raw_range[4];
   uint8_t bottom_filter[4];
   uint8_t bottom_amplitude[4];
} PD0_BottomTrackRange;

typedef struct _PD0_SensorData {
   int heading;
   uint8_t heading_status;
   uint16_t heading_source;
   int pitch;
   uint8_t pitch_status;
   uint16_t pitch_source;
   int roll;
   uint8_t roll_status;
   uint16_t roll_source;
   int speed_of_sound;
   uint8_t speed_of_sound_status;
   uint16_t speed_of_sound_source;
   int temperature;
   uint8_t temperature_status;
   uint16_t temperature_source;
   int salinity;
   uint8_t salinity_status;
   uint16_t salinity_source;
   int depth;
   uint8_t depth_status;
   uint16_t depth_source;
   int pressure;
   uint8_t pressure_status;
   uint16_t pressure_source;
   uint32_t ensemble_timer_ticks;
} PD0_SensorData;

/** Binary Data Format PD4 */
typedef struct _PD4_Data {
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
} PD4_Data;

/** Binary Data Format PD5 */
typedef struct _PD5_Data {
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
} PD5_Data;

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

/** Format for the message recieved from the DVL. */
typedef enum _MessageFormat {
   TEXT, PD0, PD4, PD5
} MessageFormat;

/** Memory storage for a message recieved from the DVL. */
typedef std::vector<char> Payload;

/** Message read/written from the hardware device. */
typedef struct _Message {
   /** Pointer to the memory allocated for this message. */
   std::shared_ptr<Payload>      payload;
   /** Pointers to portions of the PD0 payload */
   PD0_Header                    *pd0_header;
   PD0_FixedLeader               *pd0_fixed;
   PD0_VariableLeader            *pd0_variable;
   PD0_CellShortFields           *pd0_velocity;
   PD0_CellByteFields            *pd0_correlation;
   PD0_CellByteFields            *pd0_echo_intensity;
   PD0_CellByteFields            *pd0_percent_good;
   PD0_CellByteFields            *pd0_status;
   PD0_BottomTrack               *pd0_bottom_track;
   PD0_Environment               *pd0_environment;
   PD0_BottomTrackCommand        *pd0_bottom_track_command;
   PD0_BottomTrackHighRes        *pd0_bottom_track_highres;
   PD0_BottomTrackRange          *pd0_bottom_track_range;
   PD0_SensorData                *pd0_sensor_data;   
   /** Pointer to the PD4 formatted data. */
   PD4_Data                      *pd4_data;
   /** Pointer to the PD5 formatted data */
   PD5_Data                      *pd5_data;
   /** Pointer to the text */
   const char*                   text;
   /** Type of data stored in this message */
   MessageFormat                 format;
} Message;

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

typedef struct _DVLData {
   int blah;
} DVLData;

#pragma pack(pop)
