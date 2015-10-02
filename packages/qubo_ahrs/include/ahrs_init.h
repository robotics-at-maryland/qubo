#ifndef AHRS_INIT_H
#define AHRS_INIT_H

#include "ros/ros.h"

void add_reading(const boost::shared_ptr<Message const>&);

#endif
