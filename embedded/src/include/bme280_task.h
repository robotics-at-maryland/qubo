/* 
	Header file for BME280 i2c pressure, temperature, and humidity sensor
	Data sheet is here:
	 https://cdn-shop.adafruit.com/datasheets/BST-BME280_DS001-10.pdf
	Because the Tivaware libraries shift bits left to read/write the read/write
	addresses are the same
	Left most bit is considered bit 0, which is opposite of what the data
	sheet states 
*/

#include <i2c_task/i2c_task.h>

#ifndef BME280_TASK_H
#define BME280_TASK_H


#define BME280_ADDR 		0xEC /* 6th bit is changable */
#define BME280_ID			0xD0
#define BME280_CONFIG 		0xF5 
#define BME280_STATUS 		0xF3 /* bit 4 is measuring bit 7 is updating */
#define BME280_CTRL_MEAS	0xF4
#define BME280_RESET 		0xE0

/* These should not be used ofte, as we should burst read all at once */
#define BME280_HUM_LSB		0xFE
#define BME280_HUM_MSB 		0xFD
#define BME280_TEMP_XLSB 	0xFC /* only bits 0:3 used */
#define BME280_TEMP_LSB 	0xF8
#define BME280_TEMP_MSB 	0xFA
#define BME280_PRESS_XLSB 	0xF9 /* only bits 0:3 used */
#define BME280_PRESS_LSB 	0xF8
#define BME280_PRESS_MSB	0xF7



typedef struct {
	float 		pressure,
				temperature,
				humidity;
	address_t 	address;
	uint32_t 	pressure_reg,
				temperature_reg,
				humidity_reg;


} bme280;

