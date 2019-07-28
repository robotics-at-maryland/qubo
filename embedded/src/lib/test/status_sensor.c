/*
 * Nate Renegar
 * naterenegar@gmail.com
 * R@M 2019
 */

#include "../include/status_sensor.h"

void init_sensor(status_sensor *sensor) {
  sensor->data_start = sensor->buffer_start;
  sensor->data_end = sensor->buffer_start;
  sensor->buffer_end = sensor->buffer_start + STATUS_BUFFER_SIZE;
}

ssize_t write_status(status_sensor *sensor, const char *message) {
  ssize_t written_bytes = 0;
  size_t offset, distance = 0;

  sensor->data_start = sensor->data_end;

  while(*message) {
    distance = (sensor->data_end - sensor->buffer_start) % STATUS_BUFFER_SIZE;
    *(sensor->buffer_start + distance) = *message;
    message++;
    sensor->data_end++;
    written_bytes++;
  }

  /* Adjust data_end to be inside the buffer */
  offset = written_bytes % STATUS_BUFFER_SIZE;
  if(sensor->data_start + offset > sensor->data_end) {
    size_t bytes_after_start = offset - (sensor->buffer_end - sensor->data_start);
    sensor->data_end = sensor->buffer_start + bytes_after_start;
  } else {
    sensor->data_end = sensor->data_start + offset;
  }

  if(written_bytes > STATUS_BUFFER_SIZE) {
    return -1;
  } else {
    return written_bytes;
  }
}

void print_whole_buffer(status_sensor *sensor) {
  char *print_ptr = sensor->buffer_start;
  char *end_ptr = sensor->buffer_end;
  int i = 0;

  while(print_ptr != end_ptr) {
    printf("%c", *print_ptr);
    print_ptr++;
    i++;
  }

  printf("\n%d characters printed\n", i);
}

void print_valid_data(status_sensor *sensor) {
  char *print_ptr = sensor->data_start;
  char *end_ptr = sensor->data_end;
  int i = 0;

  while(print_ptr != end_ptr) {
    if(print_ptr > sensor->buffer_end) {
      print_ptr = sensor->buffer_start;
      printf("\nMoving print_ptr to buffer start\n");
    } 
    printf("%c", *print_ptr);
    print_ptr++;
    i++;
  }

  printf("\n%d characters printed\n", i);
}

