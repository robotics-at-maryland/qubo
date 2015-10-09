/*
 * read.cpp
 * Sample program for reading data from the AHRS sensor over usb.
 * Derived from the sample code from pnicorp.com
 */

#include "ahrs.h"
#include "read.h"

int main(int argc, char* argv){

}

SInt32 Setup_Sentral()
{
   UInt8 I2CTransactionStatus = 0x00;
   I2CTransactionStatus |= I2CWrite(SENTRAL_ADDRESS,ENABLE_EVENTS_REG, 0x20);// enable gyro
   // event only (for data logging function)
   I2CTransactionStatus |= I2CWrite(SENTRAL_ADDRESS,MAG_RATE_REG, 0x64);// set mag rate 100Hz
   I2CTransactionStatus |= I2CWrite(SENTRAL_ADDRESS,ACCEL_RATE_REG, 0x0a);// set accel rate 100Hz
   I2CTransactionStatus |= I2CWrite(SENTRAL_ADDRESS,GYRO_RATE_REG, 0x0f);// set gyro rate 150Hz
   I2CTransactionStatus |= I2CWrite(SENTRAL_ADDRESS,ALGORITHM_CONTROL, 0x02);// update
   I2CTransactionStatus |= I2CWrite(SENTRAL_ADDRESS,HOST_CONTROL_REG, 0x01);// Request
   // CPU to run
   if (I2CTransactionStatus)
      return RETURN_FAILURE;
   else
      return RETURN_SUCCESS;
}

SInt32 ResetSentral(UInt8 command)
{
   UInt8 ReturnedByte = 0x00, temp[1], boot_timeout = FALSE, count = 0x00;
   SInt8 ret_status = 0x00;
   ret_status = SentralRead(REVISION_ID_REG, &ReturnedByte);//read back sentral
   ROM revision, todo display over uart
      if (ret_status == TRUE) {
         I2CWrite(SENTRAL_ADDRESS,RESET_REQ_REG, 0x01);
         // Check sentral's status register to see if it has booted successfully.
         // Times out after 3 seconds.
         while (((ReturnedByte & 0x06) != 2) && (boot_timeout == FALSE)) {
            SentralRead(SENTRAL_STATUS_REG, &ReturnedByte);
            count++;
            if (count == 30)
               boot_timeout = TRUE;
            Clock_Wait(100);
         }
         if (boot_timeout) {
            PrintChars("Timeout occurred, sentral not present or took too long to
                  boot from the EEPROMn");
            return RETURN_FAILURE;
         }
         else {
            PrintChars("Boot from EEPROM successful n");
            return Setup_Sentral();
         }
      }
   PrintChars("Sentral not detectd! n");
   return RETURN_FAILURE;
}
