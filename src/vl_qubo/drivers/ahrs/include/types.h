
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
/** Identifier for configuration data */
typedef uint8_t config_id_t;
/** Identifier for sensor data */
typedef uint8_t data_id_t;

private:
/************************************************************************
 * INTERNAL STRUCT TYPEDEFS
 * Structs used for internal organization and communication.
 ************************************************************************/

/** Reference details about a command sent over serial. */
typedef struct _Command {
   /** Frame id to be sent or recieved on the serial connection. */
   frameid_t id;
   /** Size of the frame expected for this command */
   bytecount_t payload_size;
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

typedef struct _ModInfo {
   char type[4];
   uint8_t rev[4];
} ModInfo;

typedef struct _ConfigBoolean {
   config_id_t id;
   bool value;
} ConfigBoolean;

typedef struct _ConfigFloat32 {
   config_id_t id;
   float value;
} ConfigFloat32;

typedef struct _ConfigUInt8 {
   config_id_t id;
   uint8_t value;
} ConfigUInt8;

typedef struct _ConfigUInt32 {
   config_id_t id;
   uint32_t value;
} ConfigUInt32;

typedef uint8_t MagTruthMethod;

typedef uint16_t SaveError;

typedef struct _AcqParams {
   uint8_t aquisition_mode;
   uint8_t flush_filter;
   float pni_reserved;
   float sample_delay;
} AcqParams;

typedef uint32_t CalOption;

typedef uint32_t SampleCount;

typedef struct _UserCalScore {
   float mag_cal_score;
   float pni_reserved;
   float accel_cal_score;
   float distribution_error;
   float tilt_error;
   float tilt_range;
} UserCalScore;

typedef uint8_t FunctionalMode;

typedef struct _FIRFilter {
   uint8_t byte_1;
   uint8_t byte_2;
} FIRFilter;

public:
/************************************************************************
 * ENUMERATED TYPEDEFS
 * Enumerated types defining values for fields that can only take a
 * limited number of unique values.
 ************************************************************************/

/** Mounting reference ID. */
typedef enum _MountRef {
   STD_0       = 1,
   X_UP_0      = 2,
   Y_UP_0      = 3,
   STD_90      = 4,
   STD_180     = 5,
   STD_270     = 6,
   Z_DOWN_0    = 7,
   X_UP_90     = 8,
   X_UP_180    = 9,
   X_UP_270    = 10,
   Y_UP_90     = 11,
   Y_UP_180    = 12,
   Y_UP_270    = 13,
   Z_DOWN_90   = 14,
   Z_DOWN_180  = 15,
   Z_DOWN_270  = 16
} MountRef;

/** Current calibration set ID. */
typedef enum _CalibrationID {
   CAL_0, CAL_1, CAL_2, CAL_3, CAL_4, CAL_5, CAL_6, CAL_7
} CalibrationID;

/** Method for determining the truth of the magnetometer data. */
typedef enum _TruthMethod {
   STANDARD    = 1,
   TIGHT       = 2,
   AUTOMERGE   = 3
} TruthMethod;

/** Type of calibration to perform. */
typedef enum _CalType {
   FULL_RANGE = 10,
   TWO_DIM = 20,
   HARD_IRON_ONLY = 30,
   LIMITED_TILT = 40,
   ACCEL_ONLY = 100,
   MAG_AND_ACCEL = 110
} CalType;

/** FIR Filter coefficient counts. */
typedef enum _FilterCount {
   F_0   = 0,
   F_4   = 4,
   F_8   = 8,
   F_16  = 16,
   F_32  = 32
} FilterCount;

/************************************************************************
 * API TYPEDEFS
 * Structs used to communicate with the user different information.
 ************************************************************************/

/** Specification for the baudrate of the serial link. */
typedef struct _AHRSSpeed {
   /** Identifier for the device speed for the device. */
   uint8_t id;
   /** Identifier for the host speed for the computer. */
   speed_t baud;
} AHRSSpeed;

/** Struct for configuring the data aquisitions. */
typedef struct _AcqConfig {
   /** Delay between samples while in continous mode. */
   float sample_delay;
   /** true will flush the collected data every time a reading is taken. */
   bool flush_filter;
   /** true for polling aquisition, false for continuous */
   bool poll_mode;
} AcqConfig;

/** Struct containing calibration report data. */
typedef struct _CalScore {
   /** Magnetometer score (smaller is better, <2 is best). */
   float mag_score;
   /** Accelerometer score (smaller is better, <1 is best) */
   float accel_score;
   /** Distribution quality (should be 0) */
   float dist_error;
   /** Tilt angle quality (should be 0) */
   float tilt_error;
   /** Range of tilt angle (larger is better) */
   float tilt_range;
} CalScore;

/** FIR filter coefficient data with the filter count. */
typedef std::vector<double> FilterData;

/******************************************************************************
 * USER DEFINED TYPEDEFS
 * These types can be manipulated as to change the behavior of data flow.
 * Integrity and book-keeping are paramount, as many data payloads need a certain
 * format in order to be interpreted correctly.
 ******************************************************************************/

/**
 * Struct of data types being sent/retrieved from the AHRS.
 * The idCount at the beginning is how many different ids are in the struct.
 */
typedef struct _RawDataFields {
   uint8_t idCount;
   data_id_t qID;
   data_id_t gxID;
   data_id_t gyID;
   data_id_t gzID;
   data_id_t axID;
   data_id_t ayID;
   data_id_t azID;
   data_id_t mxID;
   data_id_t myID;
   data_id_t mzID;
} RawDataFields;

/**
 * Struct of data being sent/retrieved from the AHRS
 * VERY IMPORTANT: each major field must be one of the types defined
 * in the PNI AHRS spec, and must be preceded by a garbage data_id_t var.
 * This is in order to read in the data directly, and discard the data IDs.
 * Additionally, there is one garbage idCount at the beginning.
 * ALSO: make sure to update AHRS_RAW_N_FIELDS to the number of major fields.
 */
typedef struct _RawData {
   uint8_t idCount;
   data_id_t qID;
   float quaternion[4];
   data_id_t gxID;
   float gyroX;
   data_id_t gyID;
   float gyroY;
   data_id_t gzID;
   float gyroZ;
   data_id_t axID;
   float accelX;
   data_id_t ayID;
   float accelY;
   data_id_t azID;
   float accelZ;
   data_id_t mxID;
   float magX;
   data_id_t myID;
   float magY;
   data_id_t mzID;
   float magZ;
} RawData;

public:
/** Data type storing formatted AHRS data to be passed around. */
typedef struct _AHRSData {
   float quaternion[4];
   float gyroX;
   float gyroY;
   float gyroZ;
   float accelX;
   float accelY;
   float accelZ;
   float magX;
   float magY;
   float magZ;
} AHRSData;

#pragma pack(pop)
