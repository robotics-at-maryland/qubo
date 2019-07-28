/*
 * Nate Renegar
 * nrenegar@umd.edu
 * R@M 2019
 * 
 */

/* 
 * From the Power Board schematics:
 * CH0: Input Current from Battery(ies) 
 * CH1: Input Voltage from Battery(ies)
 * CH2: UWE (DVL Regulator) Current
 * CH3: 10V Regulator Current
 * CH4: 10V Regulator Voltage
 * CH5: 5V Regulator Current
 * CH6: Unused
 * CH7: Unused
 * 
 */	

#include "tasks/include/ads7828_task.h"
bool ads7828_task_init() {

	if(xTaskCreate(ads7828_task_loop, "ADS7828 TASK", 256, NULL, tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
		return true;
	}
	return false;	

}

static void ads7828_task_loop(void *params) {
	const TickType_t xDelay250ms = pdMS_TO_TICKS(250);	
	uint16_t poll = 0x0000;

	ads7828_begin(I2C0_BASE, DEFAULT_ADDRESS);	
		
	poll = ads7828_readChannel(I2C0_BASE, 0);		
	#ifdef DEBUG
	UARTprintf("polled ads7828 with value %d", poll);
	#endif
	for (;;) {
		poll = ads7828_readChannel(I2C0_BASE, 0);		

		vTaskDelay(xDelay250ms);
	}
}



