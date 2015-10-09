#include "ahrs.h"

int upload_fw(uint32_t* numericArgs, uint8_t numNumeric, float* floatArgs,
      uint8_t numFloat, char** charArgs, uint8_t numChar){
   UInt32 numBytes = 0;
   
   /* charArgs[0] must contain the name of the firmware file */
   FILE* fw = fopen(charArgs[0], "rb");
   if(!fw)
   {
      printf("Unable to open firmware image '%s'\n", charArgs[0]);
      return -1;
   }

   /* preallocate space to store the COM port name */
   char serialPort[] = "COM%d";
   /* numericArgs[0] must contain the com port number */
   sprintf(serialPort, "COM%d", numericArgs[0]);

   /* this requires an external library/classes */
   Serial* port = new Serial(serialPort, BAUD_RATE);
   if(!port->isOpened())
   {
      printf("Unable to open serial port %s\n", serialPort);
      fclose(fw);
      return -1;
   }

   /* find the length of the firmware file */
   fseek(fw, 0, SEEK_END);
   numBytes = ftell(fw);
   fseek(fw, 0, SEEK_SET);
   
   /* big struct defined in ahrs.h */
   EEPROMHeader header;

   /* read in the header from the firmware file */
   fread(&header, sizeof(header), 1, fw);

   // validate EEPROM image

   if(header.Magic != EEPROM_MAGIC_VALUE)
   {
      // magic number does not match eeprom magic.
      printf("Invalid firmware image: magic number does not match.\n");
      return -1;
   }
   if(header.EEPROMDataLength)
   {
      /* 
       * Unable to upload data via i2c, bootloader must upload by reading from eeprom.
       * NOTE: if needed, a helper uploader can be used to perform a two stage
       * upload. 
       */
      printf("Unable to upload firmware image. Firmware may only be loaded from
            EEPROM.");
      return -1;
   }

   if(header.EEPROMTextLength != numBytes - sizeof(EEPROMHeader))
   {
      /* 
       * Number of bytes remaining in file does not match the number of bytes in
       * the header. 
       */
      printf("Firmware image is incomplete. Expected %d bytes, found %d.\n",
            header.EEPROMTextLength, numBytes - sizeof(EEPROMHeader));
      return -1;
   }

   printf("Preparing to upload %d bytes...\n", numBytes - sizeof(EEPROMHeader));

   UInt32 value = i2c_read(I2C_SLAVE_ADDR, 0x37, NULL, 1, port);
   printf("SentralStatus = 0x%0.2X\n", value);
   if(value & 0x02)
   {
      printf("Sentral already has eeprom firmware loaded.\n");
   }
   /* 
    * Write value 0x01 to the ResetReq register, address 0x9B. This will result
    * in a hard reset of the Sentral. This is unnecessary if the prior event was
    * a Reset.
    */
   if(!(value & 0x08))
   {
      printf("CPU is not in standby, issuing a shutdown request.\n");
      uint32_t data[] = { 0x00 };
      i2c_write(I2C_SLAVE_ADDR, 0x34, data, 1, port);
      UInt32 value = i2c_read(I2C_SLAVE_ADDR, 0x34, NULL, 1, port);
      printf("HostControl = 0x%0.2X\n", value);
      do {
         value = i2c_read(I2C_SLAVE_ADDR, 0x37, NULL, 1, port);
         printf("SentralStatus = 0x%0.2X\n", value);
         Sleep(100);
      } while(!(value & 0x08));
   }

   printf("Enabling upload mode...\n");
   /* 
    * Write value 0x02 to the HostControl register, address 0x34. This will
    * enable an upload of the Configuration File.
    */
      uint32_t data[] = { 0x02 };
   i2c_write(I2C_SLAVE_ADDR, 0x34, data, 1, port);
   value = i2c_read(I2C_SLAVE_ADDR, 0x34, NULL, 1, port);
   printf("HostControl = 0x%0.2X\n", value);

   printf("Uploading data...\n");

#define TRASACTION_SIZE 3
   for(int i = 0; i < header.EEPROMTextLength; i += TRASACTION_SIZE * 4)
   {
      uint32_t* data = new uint32_t[TRASACTION_SIZE * 4];
      for(int j = 0; j < TRASACTION_SIZE; j++)
      {
         uint32_t value;
         fread(&value, 4, 1, fw);
         data[j * 4 + 0] = (value >> 24) & 0xFF;
         data[j * 4 + 1] = (value >> 16) & 0xFF;
         data[j * 4 + 2] = (value >> 8) & 0xFF;
         data[j * 4 + 3] = (value >> 0) & 0xFF;
      }
      if(header.EEPROMTextLength < (i + (TRASACTION_SIZE * 4)))
      {
         uint32_t bytes = header.EEPROMTextLength - i;
         i2c_write(I2C_SLAVE_ADDR, 0x96, data, bytes, port);
      }
      else
      {
         /* 
          * Write the Configuration File to Sentral.s program RAM. The file is sent
          * one byte at a time, using the UploadData register, register address 0x96.
          */

         i2c_write(I2C_SLAVE_ADDR, 0x96, data, TRASACTION_SIZE * 4, port);
      }
      delete data;
   }
   uint32_t crc[4];

   /* 
    * Read the CRC-32 register, address 0x97 . 0x9A. Compare this to the Host
    * calculated CRC-32 to confirm a successful upload.
    */
   i2c_read(I2C_SLAVE_ADDR, 0x97, crc, 4, port);
   uint32_t actualCRC = crc[0] << 0 | crc[1] << 8 | crc[2] << 16 | crc[3] <<
      24;

   if(actualCRC != header.EEPROMTextCRC)
   {
      printf("Program crc (0x%.8X) does not match CRC reported by Sentral
            (0x%0.8X)\n", header.EEPROMTextCRC, actualCRC);
   }
   else
   {
      printf("Firmware Upload Complete.\n");
   }

   fclose(fw);

   port->close();
   delete port;
   return 0;
}
