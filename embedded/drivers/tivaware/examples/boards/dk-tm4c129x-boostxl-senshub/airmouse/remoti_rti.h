//*****************************************************************************
//
// remoti_rti.h - RemoTI API defines and declarations.
//
// Copyright (c) 2014-2016 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
// 
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
// 
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
// 
// This is part of revision 2.1.3.156 of the DK-TM4C129X Firmware Package.
//
//*****************************************************************************

#ifndef __REMOTI_RTI_H__
#define __REMOTI_RTI_H__

//*****************************************************************************
//
// If building with a C++ compiler, make all of the definitions in this header
// have a C binding.
//
//*****************************************************************************
#ifdef __cplusplus
extern "C"
{
#endif

//*****************************************************************************
//
// Defines for the RTI status and error codes.
//
//*****************************************************************************
#define RTI_SUCCESS             0x00
#define RTI_ERROR_INVALID_INDEX 0xF9
#define RTI_ERROR_INVALID_PARAMETER                                           \
                                0xE8
#define RTI_ERROR_UNSUPPORTED_ATTRIBUTE                                       \
                                0xF4
#define RTI_ERROR_NO_ORG_CAPACITY                                             \
                                0xB0
#define RTI_ERROR_NO_REC_CAPACITY                                             \
                                0xB1
#define RTI_ERROR_NO_PAIRING_INDEX                                            \
                                0xB2
#define RTI_ERROR_NO_RESPONSE   0xB3
#define RTI_ERROR_NOT_PERMITTED 0xB4
#define RTI_ERROR_FRAME_COUNTER_EXPIRED                                       \
                                0xB6
#define RTI_ERROR_DISCOVERY_ERROR                                             \
                                0xB7
#define RTI_ERROR_DISCOVERY_TIMEOUT                                           \
                                0xB8
#define RTI_ERROR_SECURITY_TIMEOUT                                            \
                                0xB9
#define RTI_ERROR_SECURITY_FAILURE                                            \
                                0xBA
#define RTI_ERROR_NO_SECURITY_KEY                                             \
                                0xBD
#define RTI_ERROR_OUT_OF_MEMORY 0xBE
#define RTI_ERROR_OSAL_NO_TIMER_AVAIL                                         \
                                0x08
#define RTI_ERROR_OSAL_NV_OPER_FAILED                                         \
                                0x0A
#define RTI_ERROR_OSAL_NV_ITEM_UNINIT                                         \
                                0x09
#define RTI_ERROR_OSAL_NV_BAD_ITEM_LEN                                        \
                                0x0C
#define RTI_ERROR_MAC_TRANSACTION_EXPIRED                                     \
                                0xF0
#define RTI_ERROR_MAC_TRANSACTION_OVERFLOW                                    \
                                0xF1
#define RTI_ERROR_MAC_NO_RESOURCES                                            \
                                0x1A
#define RTI_ERROR_MAC_UNSUPPORTED                                             \
                                0x18
#define RTI_ERROR_MAC_BAD_STATE 0x19
#define RTI_ERROR_MAC_CHANNEL_ACCESS_FAILURE                                  \
                                0xE1
#define RTI_ERROR_MAC_NO_ACK    0xE9
#define RTI_ERROR_MAC_BEACON_LOST                                             \
                                0xE0
#define RTI_ERROR_MAC_PAN_ID_CONFLICT                                         \
                                0xEE
#define RTI_ERROR_MAC_SCAN_IN_PROGRESS                                        \
                                0xFC
#define RTI_ERROR_UNKNOWN_STATUS_RETURNED                                     \
                                0x20
#define RTI_ERROR_FAILED_TO_DISCOVER                                          \
                                0x21
#define RTI_ERROR_FAILED_TO_PAIR                                              \
                                0x22
#define RTI_ERROR_ALLOW_PAIRING_TIMEOUT                                       \
                                0x23
#define RTI_ERROR_FAILED_TO_CONFIGURE_ZRC                                     \
                                0x41
#define RTI_ERROR_FAILED_TO_CONFIGURE_ZID                                     \
                                0x42
#define RTI_ERROR_FAILED_TO_CONFIGURE_Z3D                                     \
                                0x43
#define RTI_ERROR_FAILED_TO_CONFIGURE_INV_MASK                                \
                                0x40
#define RTI_ERROR_SYNCHRONOUS_NPI_TIMEOUT                                     \
                                0xFF

//*****************************************************************************
//
// Define RTI_SendDataReq TX Options.
//
//*****************************************************************************
#define RTI_TX_OPTION_BROADCAST 0x01
#define RTI_TX_OPTION_IEEE_ADDRESS                                            \
                                0x02
#define RTI_TX_OPTION_ACKNOWLEDGED                                            \
                                0x04
#define RTI_TX_OPTION_SECURITY  0x08
#define RTI_TX_OPTION_SINGLE_CHANNEL                                          \
                                0x10
#define RTI_TX_OPTION_CHANNEL_DESIGNATOR                                      \
                                0x20
#define RTI_TX_OPTION_VENDOR_SPECIFIC                                         \
                                0x40

//*****************************************************************************
//
// Define RTI_ReceiveDataInd RX Flags.
//
//*****************************************************************************
#define RTI_RX_FLAGS_BROADCAST  0x01
#define RTI_RX_FLAGS_SECURITY   0x02
#define RTI_RX_FLAGS_VENDOR_SPECIFIC                                          \
                                0x04

//*****************************************************************************
//
// Define RTI_RxEnableReq parameters.
//
//*****************************************************************************
#define RTI_RX_ENABLE_OFF       0
#define RTI_RX_ENABLE_ON        0xFFFF

//*****************************************************************************
//
// RTI_StandbyReq parameters.
//
//*****************************************************************************
#define RTI_STANDBY_OFF         0
#define RTI_STANDBY_ON          1

//*****************************************************************************
//
// Define values dictated by the RF4CE standard.
//
// The following constant values are dictated by RF4CE standard and hence
// cannot be modified at all.
//*****************************************************************************
#define RTI_MAX_NUM_PROFILE_IDS 7
#define RTI_VENDOR_STRING_LENGTH                                              \
                                7
#define RTI_USER_STRING_LENGTH  15
#define RTI_SEC_KEY_SEED_LENGTH 80
#define RTI_DEST_PAN_ID_WILDCARD                                              \
                                0xFFFF
#define RTI_DEST_NWK_ADDR_WILDCARD                                            \
                                0xFFFF
#define RTI_REC_DEV_TYPE_WILDCARD                                             \
                                0xFF
#define RTI_INVALID_PAIRING_REF 0xFF
#define RTI_INVALID_DEVICE_TYPE 0xFF

//*****************************************************************************
//
// Define RTI Configuration Parameter defaults
//
// The following configuration parameter default values are set arbitrarily.
// Modifying these constant values will require rebuilding RTI module and
// probably the RNP image on the co-processor.
//
//*****************************************************************************
#if RCN_FEATURE_SECURITY
#define RTI_DEFAULT_NODE_CAPABILITIES                                         \
                                (RTI_NODE_CAP_SEC_CAP_BM | \
                                 RTI_NODE_CAP_CHAN_NORM_BM)
#else
#define RTI_DEFAULT_NODE_CAPABILITIES                                         \
                                RTI_NODE_CAP_CHAN_NORM_BM
#endif

#if FEATURE_USER_STRING_PAIRING
#define RTI_DEFAULT_APP_CAPABILITIES                                          \
                                RTI_APP_CAP_USER_STR_BM
#else
#define RTI_DEFAULT_APP_CAPABILITIES                                          \
                                0
#endif

#define RTI_STANDBY_ACTIVE_PERIOD                                             \
                                16
#define RTI_STANDBY_DUTY_CYCLE  330

//*****************************************************************************
//
// Define State Attributes (SA) Table Item Identifiers.
//
//*****************************************************************************
#define RTI_SA_ITEM_START       0x60
#define RTI_SA_ITEM_STANDBY_ACTIVE_PERIOD                                     \
                                0x60
#define RTI_SA_ITEM_CURRENT_CHANNEL                                           \
                                0x61
#define RTI_SA_ITEM_DISCOVERY_LQI_THRESHOLD                                   \
                                0x62
#define RTI_SA_ITEM_DUTY_CYCLE  0x64
#define RTI_SA_ITEM_FRAME_COUNTER                                             \
                                0x65
#define RTI_SA_ITEM_IN_POWER_SAVE                                             \
                                0x67
#define RTI_SA_ITEM_MAX_FIRST_ATTEMPT_CSMA_BACKOFFS                           \
                                0x6A
#define RTI_SA_ITEM_MAX_FIRST_ATTEMPT_FRAME_RETRIES                           \
                                0x6B
#define RTI_SA_ITEM_RESPONSE_WAIT_TIME                                        \
                                0x6D
#define RTI_SA_ITEM_SCAN_DURATION                                             \
                                0x6E
#define RTI_SA_ITEM_USER_STRING 0x6F
#define RTI_SA_ITEM_PAN_ID      0x85
#define RTI_SA_ITEM_SHORT_ADDRESS                                             \
                                0x86
#define RTI_SA_ITEM_AGILITY_ENABLE                                            \
                                0x87
#define RTI_SA_ITEM_TRANSMIT_POWER                                            \
                                0x88

//*****************************************************************************
//
// Define Configuration Parameters (CP) Table Item Identifiers
//
//*****************************************************************************
#define RTI_CP_ITEM_START       0xA0
#define RTI_CP_ITEM_STARTUP_CTRL                                              \
                                0xA0
#define RTI_CP_ITEM_NODE_CAPABILITIES                                         \
                                0xA1
#define RTI_CP_ITEM_NODE_SUPPORTED_TGT_TYPES                                  \
                                0xA2
#define RTI_CP_ITEM_APPL_CAPABILITIES                                         \
                                0xA3
#define RTI_CP_ITEM_APPL_DEV_TYPE_LIST                                        \
                                0xA4
#define RTI_CP_ITEM_APPL_PROFILE_ID_LIST                                      \
                                0xA5
#define RTI_CP_ITEM_STDBY_DEFAULT_ACTIVE_PERIOD                               \
                                0xA6
#define RTI_CP_ITEM_STDBY_DEFAULT_DUTY_CYCLE                                  \
                                0xA7
#define RTI_CP_ITEM_VENDOR_ID   0xA8
#define RTI_CP_ITEM_VENDOR_NAME 0xA9
#define RTI_CP_ITEM_DISCOVERY_DURATION                                        \
                                0xAA
#define RTI_CP_ITEM_DEFAULT_DISCOVERY_LQI_THRESHOLD                           \
                                0xAB
#define RTI_CP_ITEM_END         0xAF

//*****************************************************************************
//
// Define Pairing Table constants.
//
//*****************************************************************************
#define RTI_SA_ITEM_PT_NUMBER_OF_ACTIVE_ENTRIES                               \
                                0xB0
#define RTI_SA_ITEM_PT_CURRENT_ENTRY_INDEX                                    \
                                0xB1
#define RTI_SA_ITEM_PT_CURRENT_ENTRY                                          \
                                0xB2
#define RTI_SA_ITEM_PT_END      0xB3

//*****************************************************************************
//
// Define Constants (CONST) Table Item Identifiers
//
//*****************************************************************************
#define RTI_CONST_ITEM_START    0xC0
#define RTI_CONST_ITEM_SW_VERSION                                             \
                                0xC0
#define RTI_CONST_ITEM_MAX_PAIRING_TABLE_ENTRIES                              \
                                0xC1
#define RTI_CONST_ITEM_NWK_PROTOCOL_IDENTIFIER                                \
                                0xC2
#define RTI_CONST_ITEM_NWK_PROTOCOL_VERSION                                   \
                                0xC3
#define RTI_CONST_ITEM_END      0xC4

//*****************************************************************************
//
// Miscellaneous constants
//
//*****************************************************************************
#define RTI_MAX_NUM_DEV_TYPES   3           // number of device types supported
#define RTI_SEC_KEY_LENGTH      16          // length of security key in bytes
#define RTI_MAX_NUM_SUPPORTED_TGT_TYPES                                       \
                                6

//*****************************************************************************
//
//! Startup Control enum values. These values are used to govern how the Remote
//! network processor will startup and what will be preserved or cleared.
//
//*****************************************************************************
enum
{
    //
    //! Restore all state information.  Pairing data and status is kept.
    //
    eRTI_RESTORE_STATE,

    //
    //! Clear the state data but keep configuration values.  Pairing status is
    //! lost.
    //
    eRTI_CLEAR_STATE,

    //
    //! Clear all configuration and state information from the RNP.
    //
    eRTI_CLEAR_CONFIG_CLEAR_STATE
};

//*****************************************************************************
//
// Extra pairing entry information feature compile flag.
// Extra pairing entry information pertains to vendor ID and device type list.
// Disabling this feature reduces the pairing table size but
// note that feature flag definitioin has to match the RCN library
// that the application is compiled with.
// That is, both application code and the RCN library must have been compiled with
// the same feature selection.
//
// Changing this setting is NOT recommended.
//
//*****************************************************************************
#ifndef RTI_FEATURE_EXTRA_PAIR_INFO
#define RTI_FEATURE_EXTRA_PAIR_INFO                                           \
                                1
#endif

//*****************************************************************************
//
// Discrete bit definitions for rcnNwkPairingEntry_t.profileDiscs[].
//
//*****************************************************************************
#if RTI_FEATURE_EXTRA_PAIR_INFO

#define RTI_PROFILE_DISC_GT8    0           // GT8 set means one or more >= 8.
#define RTI_PROFILE_DISC_ZRC    1
#define RTI_PROFILE_DISC_ZID    2
#define RTI_PROFILE_DISC_Z3D    3
#define RTI_PROFILE_DISC_SP4    4
#define RTI_PROFILE_DISC_SP5    5
#define RTI_PROFILE_DISC_SP6    6
#define RTI_PROFILE_DISC_SP7    7

#define RTI_PROFILE_DISCS_SIZE  1           // Profile Discretes bytes allowed.

#endif

//*****************************************************************************
//
// Typedef pairing table entry.
//
// \note The RF4CE spec does not include \bui16VendorIdentifier,
// \bpui8DevTypeList and \bpui8ProfileDiscs as part of pairing entry.
//
//*****************************************************************************
typedef struct
{
    //
    // Pairing reference
    //
    uint8_t  ui8PairingRef;

    //
    // This device's own short address.
    //
    uint16_t ui16SrcNwkAddress;

    //
    // Expected RF channel of the peer.
    //
    uint8_t  ui8LogicalChannel;

    //
    // IEEE address of the peer.
    //
    uint8_t  pui8IEEEAddress[8];

    //
    // PAN identifier of the peer.
    //
    uint16_t ui16panId;

    //
    // Network address of the peer.
    //
    uint16_t ui16NwkAddress;

    //
    // Capabilities bitmap of the target.
    //
    uint8_t  ui8RecCapabilities;

    //
    // Whether the key is valid or not.
    //
    uint8_t  ui8SecurityKeyValid;

    //
    // The actual security key for this pariing.
    //
    uint8_t  pui8SecurityKey[RTI_SEC_KEY_LENGTH]; // security link key

#if RTI_FEATURE_EXTRA_PAIR_INFO
    //
    // Vendor identification number of the target.
    //
    uint16_t ui16VendorIdentifier;

    //
    // list of device types supported by the peer
    //
    uint8_t  pui8DevTypeList[RTI_MAX_NUM_DEV_TYPES];

#endif // RTI_FEATURE_EXTRA_PAIR_INFO

    //
    // Frame counter of last received message.
    //
    uint32_t ui32FrameCounter;

#if RTI_FEATURE_EXTRA_PAIR_INFO
    //
    // Profile Discovery
    //
    uint8_t     profileDiscs[RTI_PROFILE_DISCS_SIZE];

#endif
} tNWKPairingEntry;

//*****************************************************************************
//
// Device Type List
//
//*****************************************************************************
#define RTI_DEVICE_RESERVED_INVALID                                           \
                                0x00
#define RTI_DEVICE_REMOTE_CONTROL                                             \
                                0x01
#define RTI_DEVICE_TARGET_TYPE_START                                          \
                                0x02
#define RTI_DEVICE_TELEVISION   0x02
#define RTI_DEVICE_PROJECTOR    0x03
#define RTI_DEVICE_PLAYER       0x04
#define RTI_DEVICE_RECORDER     0x05
#define RTI_DEVICE_VIDEO_PLAYER_RECORDER                                      \
                                0x06
#define RTI_DEVICE_AUDIO_PLAYER_RECORDER                                      \
                                0x07
#define RTI_DEVICE_AUDIO_VIDEO_RECORDER                                       \
                                0x08
#define RTI_DEVICE_SET_TOP_BOX  0x09
#define RTI_DEVICE_HOME_THEATER_SYSTEM                                        \
                                0x0A
#define RTI_DEVICE_MEDIA_CENTER_PC                                            \
                                0x0B
#define RTI_DEVICE_GAME_CONSOLE 0x0C
#define RTI_DEVICE_SATELLITE_RADIO_RECEIVER                                   \
                                0x0D
#define RTI_DEVICE_IR_EXTENDER  0x0E
#define RTI_DEVICE_MONITOR      0x0F
#define RTI_DEVICE_TARGET_TYPE_END                                            \
                                0x10

//
// 0x10..0xFD: Reserved
//
#define RTI_DEVICE_GENERIC      0xFE
#define RTI_DEVICE_RESERVED_FOR_WILDCARDS                                     \
                                0xFF

#define MAX_AVAIL_DEVICE_TYPES (RTI_DEVICE_TARGET_TYPE_END - \
                                RTI_DEVICE_TARGET_TYPE_START)

//*****************************************************************************
//
// vendor identifiers
//
//*****************************************************************************
#define RTI_VENDOR_PANASONIC    0x0001
#define RTI_VENDOR_SONY         0x0002
#define RTI_VENDOR_SAMSUNG      0x0003
#define RTI_VENDOR_PHILIPS      0x0004
#define RTI_VENDOR_FREESCALE    0x0005
#define RTI_VENDOR_OKI          0x0006
#define RTI_VENDOR_TEXAS_INSTRUMENTS                                          \
                                0x0007

//*****************************************************************************
//
// profile identifiers
//
//*****************************************************************************
#define RTI_PROFILE_RTI         0xFF
#define RTI_PROFILE_ID_START    0x01
#define RTI_PROFILE_GDP         0x00
#define RTI_PROFILE_ZRC         0x01
#define RTI_PROFILE_ZID         0x02
#define RTI_PROFILE_Z3S         0x03
#define RTI_PROFILE_ID_END      0x04
#define RTI_PROFILE_TI          0xC0

//*****************************************************************************
//
// CERC command codes
//
//*****************************************************************************
#define RTI_CERC_USER_CONTROL_PRESSED                                         \
                                0x01
#define RTI_CERC_USER_CONTROL_REPEATED                                        \
                                0x02
#define RTI_CERC_USER_CONTROL_RELEASED                                        \
                                0x03
#define RTI_CERC_COMMAND_DISCOVERY_REQUEST                                    \
                                0x04
#define RTI_CERC_COMMAND_DISCOVERY_RESPONSE                                   \
                                0x05
#define RTI_CERC_SELECT         0x00
#define RTI_CERC_UP             0x01
#define RTI_CERC_DOWN           0x02
#define RTI_CERC_LEFT           0x03
#define RTI_CERC_RIGHT          0x04
#define RTI_CERC_RIGHT_UP       0x05
#define RTI_CERC_RIGHT_DOWN     0x06
#define RTI_CERC_LEFT_UP        0x07
#define RTI_CERC_LEFT_DOWN      0x08
#define RTI_CERC_ROOT_MENU      0x09
#define RTI_CERC_SETUP_MENU     0x0A
#define RTI_CERC_CONTENTS_MENU  0x0B
#define RTI_CERC_FAVORITE_MENU  0x0C
#define RTI_CERC_EXIT           0x0D
#define RTI_CERC_NUM_11         0x1e
#define RTI_CERC_NUM_12         0x1f
#define RTI_CERC_NUM_0          0x20
#define RTI_CERC_NUM_1          0x21
#define RTI_CERC_NUM_2          0x22
#define RTI_CERC_NUM_3          0x23
#define RTI_CERC_NUM_4          0x24
#define RTI_CERC_NUM_5          0x25
#define RTI_CERC_NUM_6          0x26
#define RTI_CERC_NUM_7          0x27
#define RTI_CERC_NUM_8          0x28
#define RTI_CERC_NUM_9          0x29
#define RTI_CERC_DOT            0x2A
#define RTI_CERC_ENTER          0x2B
#define RTI_CERC_CLEAR          0x2C
#define RTI_CERC_NEXT_FAVORITE  0x2F
#define RTI_CERC_CHANNEL_UP     0x30
#define RTI_CERC_CHANNEL_DOWN   0x31
#define RTI_CERC_PREVIOUS_CHANNEL                                             \
                                0x32
#define RTI_CERC_SOUND_SELECT   0x33
#define RTI_CERC_INPUT_SELECT   0x34
#define RTI_CERC_DISPLAY_INFORMATION                                          \
                                0x35
#define RTI_CERC_HELP           0x36
#define RTI_CERC_PAGE_UP        0x37
#define RTI_CERC_PAGE_DOWN      0x38
#define RTI_CERC_POWER          0x40
#define RTI_CERC_VOLUME_UP      0x41
#define RTI_CERC_VOLUME_DOWN    0x42
#define RTI_CERC_MUTE           0x43
#define RTI_CERC_PLAY           0x44
#define RTI_CERC_STOP           0x45
#define RTI_CERC_PAUSE          0x46
#define RTI_CERC_RECORD         0x47
#define RTI_CERC_REWIND         0x48
#define RTI_CERC_FAST_FORWARD   0x49
#define RTI_CERC_EJECT          0x4A
#define RTI_CERC_FORWARD        0x4B
#define RTI_CERC_BACKWARD       0x4C
#define RTI_CERC_STOP_RECORD    0x4D
#define RTI_CERC_PAUSE_RECORD   0x4E
#define RTI_CERC_ANGLE          0x50
#define RTI_CERC_SUB_PICTURE    0x51
#define RTI_CERC_VIDEO_ON_DEMAND                                              \
                                0x52
#define RTI_CERC_ELECTRONIC_PROGRAM_GUIDE                                     \
                                0x53
#define RTI_CERC_TIMER_PROGRAMMING                                            \
                                0x54
#define RTI_CERC_INITIAL_CONFIGURATION                                        \
                                0x55
#define RTI_CERC_PLAY_FUNCTION  0x60
#define RTI_CERC_PAUSE_PLAY_FUNCTION                                          \
                                0x61
#define RTI_CERC_RECORD_FUNCTION                                              \
                                0x62
#define RTI_CERC_PAUSE_RECORD_FUNCTION                                        \
                                0x63
#define RTI_CERC_STOP_FUNCTION  0x64
#define RTI_CERC_MUTE_FUNCTION  0x65
#define RTI_CERC_RESTORE_VOLUME_FUNCTION                                      \
                                0x66
#define RTI_CERC_TUNE_FUNCTION  0x67
#define RTI_CERC_SELECT_MEDIA_FUNCTION                                        \
                                0x68
#define RTI_CERC_SELECT_AV_INPUT_FUNCTION                                     \
                                0x69
#define RTI_CERC_SELECT_AUDIO_INPUT_FUNCTION                                  \
                                0x6A
#define RTI_CERC_POWER_TOGGLE_FUNCTION                                        \
                                0x6B
#define RTI_CERC_POWER_OFF_FUNCTION                                           \
                                0x6C
#define RTI_CERC_POWER_ON_FUNCTION                                            \
                                0x6D
#define RTI_CERC_F1_BLUE        0x71
#define RTI_CERC_F2_RED         0x72
#define RTI_CERC_F3_GREEN       0x73
#define RTI_CERC_F4_YELLOW      0x74
#define RTI_CERC_F5             0x75
#define RTI_CERC_DATA           0x76
#define RTI_CERC_RESERVED_1     0xFF
#define RTI_CERC_EXTENDED_COMMAND(_cmd)                                       \
  ((_cmd) == RTI_CERC_PLAY_FUNCTION)
#define RTI_CERC_SELECTION_COMMAND(_cmd)                                      \
            (((_cmd) == RTI_CERC_SELECT_MEDIA_FUNCTION) ||                    \
             ((_cmd) == RTI_CERC_SELECT_AV_INPUT_FUNCTION) ||                 \
             ((_cmd) == RTI_CERC_SELECT_AUDIO_INPUT_FUNCTION))
#define RTI_CERC_COMPOSITE_COMMAND(_cmd)                                      \
                                ((_cmd) == RTI_CERC_TUNE_FUNCTION)
#define RTI_CERC_PLAY_MODE_PLAY_FORWARD                                       \
                                0x24
#define RTI_CERC_PLAY_MODE_PLAY_REVERSE                                       \
                                0x20
#define RTI_CERC_PLAY_MODE_PLAY_STILL                                         \
                                0x25
#define RTI_CERC_PLAY_MODE_FAST_FORWARD_MINIMUM_SPEED                         \
                                0x05
#define RTI_CERC_PLAY_MODE_FAST_FORWARD_MEDIUM_SPEED                          \
                                0x06
#define RTI_CERC_PLAY_MODE_FAST_FORWARD_MAXIMUM_SPEED                         \
                                0x07
#define RTI_CERC_PLAY_MODE_FAST_REVERSE_MINIMUM_SPEED                         \
                                0x09
#define RTI_CERC_PLAY_MODE_FAST_REVERSE_MEDIUM_SPEED                          \
                                0x0A
#define RTI_CERC_PLAY_MODE_FAST_REVERSE_MAXIMUM_SPEED                         \
                                0x0B
#define RTI_CERC_PLAY_MODE_SLOW_FORWARD_MINIMUM_SPEED                         \
                                0x15
#define RTI_CERC_PLAY_MODE_SLOW_FORWARD_MEDIUM_SPEED                          \
                                0x16
#define RTI_CERC_PLAY_MODE_SLOW_FORWARD_MAXIMUM_SPEED                         \
                                0x17
#define RTI_CERC_PLAY_MODE_SLOW_REVERSE_MINIMUM_SPEED                         \
                                0x19
#define RTI_CERC_PLAY_MODE_SLOW_REVERSE_MEDIUM_SPEED                          \
                                0x1A
#define RTI_CERC_PLAY_MODE_SLOW_REVERSE_MAXIMUM_SPEED                         \
                                0x1B

//*****************************************************************************
//
// TI vendor specific command protocol identifiers
//
//*****************************************************************************
#define RTI_PROTOCOL_POLL       0x00
#define RTI_PROTOCOL_OAD        0x10
#define RTI_PROTOCOL_TEST       0x20
#define RTI_PROTOCOL_EXT_TV_RC  0x30
#define RTI_PROTOCOL_PLAY_LIST_DOWNLOAD                                       \
                                0x40

//
// TI test protocol command identifiers (use with RTI_PROTOCOL_TEST).
//
#define RTI_CMD_TEST_PARAMETERS 0x00
#define RTI_CMD_TEST_DATA       0x01
#define RTI_CMD_TEST_REPORT     0x02
#define RTI_CMD_TEST_DATA_SEQUENCED                                           \
                                0x03

//
// TI extended TV remote control command identifiers (use with
// RTI_PROTOCOL_EXT_TV_RC)
//
#define RTI_CMD_EXT_TV_ZOOM_IN  0x00
#define RTI_CMD_EXT_TV_ZOOM_OUT 0x01
#define RTI_CMD_EXT_TV_PIP_TOGGLE                                             \
                                0x02
#define RTI_CMD_EXT_TV_PIP_SWAP 0x03
#define RTI_CMD_EXT_TV_PIP_MOVE 0x04
#define RTI_CMD_EXT_TV_PIP_SIZE 0x05
#define RTI_CMD_EXT_TV_PIP_CH_DOWN                                            \
                                0x06
#define RTI_CMD_EXT_TV_PIP_CH_UP                                              \
                                0x07
#define RTI_CMD_EXT_TV_PIP_FREEZE                                             \
                                0x08

//*****************************************************************************
//
// Mark the end of the C bindings section for C++ compilers.
//
//*****************************************************************************
#ifdef __cplusplus
}
#endif

#endif // __REMOTI_RTI_H__
