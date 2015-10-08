/******************** (C) COPYRIGHT 2014 PNI Sensor Corp **************************
 * * File Name : EEPROMImage.h
 * * Date : 28-Jan-2014
 * * Description : Firmware file description and structures
 * **********************************************************************************/
Typedef UInt16 EEPROMMagic;
#define EEPROM_MAGIC_VALUE 0x652A
/** EEPROM boot flags. */
Typedef union {
   /** Direct access to all flags. */
   UInt16 value;
   Struct {
      /** Do not execute the EEPROM image immediately after upload. */
      UInt16 EEPROMNoExec:1;
      /** Reserved */
      UInt16 Reserved:7;
      /** The clock speed for uploading the firmware. */
      UInt16 I2CClockSpeed:3;
      /** The Expected Rom Version for the image. */
      UInt16 ROMVerExp:4;
      /** Reserved */
      UInt16 reserved1:1;
   } bits;
} EEPROMFlags;
#define EXP_ROM_VERSION_ANY 0x00
#define EXP_ROM_VERSION_DI01 0x01
#define EXP_ROM_VERSION_DI02 0x02
Typedef UInt32 EEPROMTextCRC;
Typedef UInt32 EEPROMDataCRC;
Typedef UInt16 EEPROMTextLength;
Typedef UInt16 EEPROMDataLength;
Typedef UInt8* EEPROMText;
Typedef UInt8* EEPROMData;
/** EEPROM header format
 * * NOTE: a ROM version may also be useful to ensure an incorrect ram binary is
 * not used.
 * * This is currently not implimented, however the RAM / EEPROM start code can
 * double check this before it starts if needed.
 * */
Typedef struct {
   /** The firmware magic number */
   UInt16 Magic; // Already read
   /** Flags used to notify and control the boot process */
   UInt16 EEPROMFlags;
   /** The CRC32-CCITT of the firmware text segment */
   UInt32 EEPROMTextCRC; // CRC32-CCITT
   /** The CRC32-CCITT of the firmware data segment */
   UInt32 EEPROMDataCRC; // CRC32-CCITT
   /** The number of program bytes to upload */
   UInt16 EEPROMTextLength;
   /** The number of data bytes to upload */
   UInt16 EEPROMDataLength;
} EEPROMHeader;
