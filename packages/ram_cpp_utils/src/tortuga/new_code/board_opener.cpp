
#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"
#include <sstream>
#include <thread>
#include "tortuga/imuapi.h"
#include "tortuga/sensorapi.h"

bool checkError(int, char*);

// This is the code used to open and publish all of the 
// file descriptors for tortuga device boards


std::string imu_file = "/dev/magboom";
std::string sensor_file = "/dev/sensor";

int main(int argc, char **argv){

	ros::init(argc, argv, "board_opener");

	//imu
	if(ros::param::has(imu_file)){
		ros::param::get("imu_device_file", imu_file);
	}else{
		ros::param::set("imu_device_file", imu_file);
	}

	int fd = openIMU(imu_file.c_str());
	checkError(fd, "IMU");

	ros::param::set("imu_file_descriptor", fd);

	//sensor board
	if(ros::param::has(sensor_file)){
		ros::param::get("sensor_board_file", sensor_file);
	}else{
		ros::param::set("sensor_board_file", sensor_file);
	}

	fd = openSensorBoard(sensor_file.c_str());
	checkError(fd, "Sensor Board");

	ros::param::set("sensor_board_file_descriptor", fd);

}



bool checkError(int e, char *name) {
    switch(e) {
    case SB_IOERROR:
      ROS_DEBUG("IO ERROR in %s", name);
      return true;
    case SB_BADCC:
      ROS_DEBUG("BAD CC ERROR in %s", name);
      return true;
    case SB_HWFAIL:
      ROS_DEBUG("HW FAILURE ERROR in %s", name);
      return true;
    case SB_ERROR:
      ROS_DEBUG("ERROR in %s", name);
      return true;
    default:
      return false;
    }
}