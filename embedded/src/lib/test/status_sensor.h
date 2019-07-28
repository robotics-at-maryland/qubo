/*
 * Nate Renegar
 * naterenegar@gmail.com
 * R@M 2019
 *
 */

/*
 * This is a ring buffer that stores error messages
 * that other sensors in the embedded system generate,
 * hence the name status sensor.
 */

#ifndef STATUS_SENSOR_H
#define STATUS_SENSOR_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define STATUS_BUFFER_SIZE 256

typedef struct _status_sensor {
  char buffer_start[STATUS_BUFFER_SIZE];
  char *data_start, *data_end, *buffer_end;
} status_sensor;

void init_sensor(status_sensor *sensor);
ssize_t write_status(status_sensor *sensor, const char *message);
void print_whole_buffer(status_sensor *sensor);
void print_valid_data(status_sensor *sensor); 

#endif
