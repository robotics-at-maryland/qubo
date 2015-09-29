#ifndef DEPTH_SENSOR_SIM_HEADER
#define DEPTH_SENSOR_SIM_HEADER

#include "ros/ros.h"
#include "underwater_sensor_msgs/Pressure.h"

void depthCallback(const underwater_sensor_msgs::Pressure msg);

#endif
