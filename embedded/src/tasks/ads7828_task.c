/*
 * Nate Renegar
 * nrenegar@cs.umd.edu
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
 */	

#include "tasks/incldue/ads7828_task.h"

bool ads7828_task_init() {

	if(xTaskCreate(ads7828_task_loop, "ADS7828 TASK", 256, NULL, tskIDLE_PRIORITY + 1, NULL) != pdTRUE) {
		return true;
	}
	return false;	

}

static void ads7828_task_loop(void *params) {



}
