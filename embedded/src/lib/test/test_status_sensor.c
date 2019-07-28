/* Unit tests for status_sensor */

#include <stdio.h>
#include <stdlib.h>

#include "../include/status_sensor.h"

int main(void) {
	status_sensor sensor1;

	/* Test inputs to status_sensor */
	char *full_length_input;
	char *length8 = "88888888";

	if(full_length_input  = malloc(STATUS_BUFFER_SIZE)) {
		memset(full_length_input, 'a', STATUS_BUFFER_SIZE-1); 
		full_length_input[STATUS_BUFFER_SIZE-1] = '\0';
	} else {
		printf("Memory allocation for test inputs failed. Exiting...\n");
		return 0;	
	}

	init_sensor(&sensor1);

	write_status(&sensor1, full_length_input);
	print_whole_buffer(&sensor1);
	
	write_status(&sensor1, length8);
	print_whole_buffer(&sensor1);	
	print_valid_data(&sensor1);	

	return(0);

}
