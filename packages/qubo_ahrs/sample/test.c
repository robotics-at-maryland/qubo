#include <stdio.h>
#include <unistd.h>
#include "imuapi.h"

int main(){
   int imu = openIMU("/dev/ttyUSB0");
   RawIMUData data; 
   if (imu == -1)
      return -1;
   printf("Connected with file descriptor %d",imu);
   while (1){
      printf("READING\n");
      readIMUData(imu,&data);
      printf("ID: %d, Timer: %d", data.messageID, data.sampleTimer);
      printf("Gyro rad/s %f/%f/%f", data.gyroX, data.gyroY, data.gyroZ);
      printf("Accel Gs %f/%f/%f", data.accelX, data.accelY, data.accelZ);
      printf("Mag gauss %f/%f/%f", data.magX, data.magY, data.magZ);
      printf("Temp rad/s %f/%f/%f", data.tempX, data.tempY, data.tempZ);
      printf("Checksum: %d\n",data.checksumValid);
   }
   close(imu);
   return 0;
}
