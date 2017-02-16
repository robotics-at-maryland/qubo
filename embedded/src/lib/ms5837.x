/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

// ported code from https://github.com/bluerobotics/BlueRobotics_MS5837_Library/

#include "lib/include/ms5837.h"

void ms5837_init(uint32_t device) {
	// Reset the MS5837, per datasheet
  // Just use one byte of the buffer here, reuse it down below
  uint8_t reg;
  uint8_t buffer[2];
  reg = MS5837_RESET;
  writeI2C(device, MS5837_ADDR, &reg, 1, false);

	// Wait for reset to complete
  vTaskDelay(10 / portTICK_RATE_MS);

	// Read calibration values and CRC
  for( uint8_t i = 0; i < 8; i++ ) {
    reg = MS5837_PROM_READ+i*2;

    queryI2C(device, MS5837_ADDR, &reg, 1, buffer, 2);

    // Not sure which should be first
    C[i] = (buffer[0] << 8) | buffer[1];
  }

	// Verify that data is correct with CRC
	uint8_t crcRead = C[0] >> 12;
	uint8_t crcCalculated = crc4(C);

	if ( crcCalculated == crcRead ) {
		// Success
	} else {
		// Failure - try again?
	}
}

void ms5837_setModel(uint8_t model) {
	_model = model;
}

void ms5837_setFluidDensity(uint32_t device, float density) {
	fluidDensity = density;
}

void ms5837_read(uint32_t device) {

  uint8_t reg = MS5837_CONVERT_D1_8192;

	// Request D1 conversion
  writeI2C(device, MS5837_ADDR, &reg, 1, false);

  vTaskDelay(20 / portTICK_RATE_MS); // Max conversion time per datasheet

  reg = MS5837_ADC_READ;
  uint8_t buffer[3];

  queryI2C(device, MS5837_ADDR, &reg, 1, buffer, 3);

	D1 = 0;
	D1 = buffer[0];
  D1 = (D1 << 8) | buffer[1];
	D1 = (D1 << 8) | buffer[2];

  reg = MS5837_CONVERT_D2_8192;

  writeI2C(device, MS5837_ADDR, &reg, 1, false);

  // Max conversion time per datasheet
  vTaskDelay(20 / portTICK_RATE_MS);

  reg = MS5837_ADC_READ;
  queryI2C(device, MS5837_ADDR, &reg, 1, buffer, 3);

	D2 = 0;
	D2 = buffer[0];
  D2 = (D2 << 8) | buffer[1];
	D2 = (D2 << 8) | buffer[2];

	calculate();
}

void ms5837_readTestCase(uint32_t device) {
	if ( _model == MS5837_02BA ) {
		C[0] = 0;
		C[1] = 46372;
		C[2] = 43981;
		C[3] = 29059;
		C[4] = 27842;
		C[5] = 31553;
		C[6] = 28165;
		C[7] = 0;

		D1 = 6465444;
		D2 = 8077636;
	} else {
		C[0] = 0;
		C[1] = 34982;
		C[2] = 36352;
		C[3] = 20328;
		C[4] = 22354;
		C[5] = 26646;
		C[6] = 26146;
		C[7] = 0;

		D1 = 4958179;
		D2 = 6815414;
	}

	calculate();
}


float ms5837_pressure(uint32_t device, float conversion) {
	return P*conversion;
}

float ms5837_temperature(uint32_t device) {
	return TEMP/100.0f;
}

float ms5837_depth(uint32_t device) {
	return (ms5837_pressure(device, Pa)-101300)/(fluidDensity*9.80665);
}

float ms5837_altitude(uint32_t device) {
	return (1-pow((ms5837_pressure(device, 1.0f)/1013.25),.190284))*145366.45*.3048;
}

static void calculate() {
	// Given C1-C6 and D1, D2, calculated TEMP and P
	// Do conversion first and then second order temp compensation

	int32_t dT = 0;
	int64_t SENS = 0;
	int64_t OFF = 0;
	int32_t SENSi = 0;
	int32_t OFFi = 0;
	int32_t Ti = 0;
	int64_t OFF2 = 0;
	int64_t SENS2 = 0;

	// Terms called
	dT = D2-(uint32_t)(C[5])*256l;
	if ( _model == MS5837_02BA ) {
		SENS = (int64_t)(C[1])*65536l+((int64_t)(C[3])*dT)/128l;
		OFF = (int64_t)(C[2])*131072l+((int64_t)(C[4])*dT)/64l;
		P = (D1*SENS/(2097152l)-OFF)/(32768l);
	} else {
		SENS = (int64_t)(C[1])*32768l+((int64_t)(C[3])*dT)/256l;
		OFF = (int64_t)(C[2])*65536l+((int64_t)(C[4])*dT)/128l;
		P = (D1*SENS/(2097152l)-OFF)/(8192l);
	}

	// Temp conversion
	TEMP = 2000l+(int64_t)(dT)*C[6]/8388608LL;

	//Second order compensation
	if ( _model == MS5837_02BA ) {
		if((TEMP/100)<20){         //Low temp
			//Serial.println("here");
			Ti = (11*(int64_t)(dT)*(int64_t)(dT))/(34359738368LL);
			OFFi = (31*(TEMP-2000)*(TEMP-2000))/8;
			SENSi = (63*(TEMP-2000)*(TEMP-2000))/32;
		}
	} else {
		if((TEMP/100)<20){         //Low temp
			Ti = (3*(int64_t)(dT)*(int64_t)(dT))/(8589934592LL);
			OFFi = (3*(TEMP-2000)*(TEMP-2000))/2;
			SENSi = (5*(TEMP-2000)*(TEMP-2000))/8;
			if((TEMP/100)<-15){    //Very low temp
				OFFi = OFFi+7*(TEMP+1500l)*(TEMP+1500l);
				SENSi = SENSi+4*(TEMP+1500l)*(TEMP+1500l);
			}
		}
		else if((TEMP/100)>=20){    //High temp
			Ti = 2*(dT*dT)/(137438953472LL);
			OFFi = (1*(TEMP-2000)*(TEMP-2000))/16;
			SENSi = 0;
		}
	}

	OFF2 = OFF-OFFi;           //Calculate pressure and temp second order
	SENS2 = SENS-SENSi;

	if ( _model == MS5837_02BA ) {
		TEMP = (TEMP-Ti);
		P = (((D1*SENS2)/2097152l-OFF2)/32768l)/100;
	} else {
		TEMP = (TEMP-Ti);
		P = (((D1*SENS2)/2097152l-OFF2)/8192l)/10;
	}
}

static uint8_t crc4(uint16_t *n_prom) {
	uint16_t n_rem = 0;

	*n_prom = ((*n_prom) & 0x0FFF);
	*(n_prom+7) = 0;

	for ( uint8_t i = 0 ; i < 16; i++ ) {
		if ( i%2 == 1 ) {
			n_rem ^= (uint16_t)((*(n_prom+(i>>1))) & 0x00FF0);
		} else {
			n_rem ^= (uint16_t)(*(n_prom+(i>>1)) >> 8);
		}
		for ( uint8_t n_bit = 8 ; n_bit > 0 ; n_bit-- ) {
			if ( n_rem & 0x8000 ) {
				n_rem = (n_rem << 1) ^ 0x3000;
			} else {
				n_rem = (n_rem << 1);
			}
		}
	}

	n_rem = ((n_rem >> 12) & 0x000F);

	return n_rem ^ 0x00;
}
