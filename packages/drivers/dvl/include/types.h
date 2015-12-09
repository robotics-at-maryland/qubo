
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
