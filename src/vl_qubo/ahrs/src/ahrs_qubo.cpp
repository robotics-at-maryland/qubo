#include "ahrs_qubo.h"
//written by Jeremy Weed

AhrsQuboNode::AhrsQuboNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string name,std::string device) : RamNode(n){

	ros::Time::init();
	this->name = name;
	ros::Rate loop_rate(rate);

	ahrsPub = n->advertise<sensor_msgs::Imu>("qubo/ahrs/" + name, 1000);

	ahrs->openDevice();
}
