#ifndef AHRS_QUBO_HEADER
#define AHRS_QUBO_HEADER
//written by Jeremy Weed

#include "ram_node.h"
#include "AHRS.h"
#include "sensor_msgs/Imu.h"



class AhrsQuboNode : public RamNode {

public:
	AhrsQuboNode(std::shared_ptr<ros::NodeHandle>,int, std::string name,
		std::string device);
	~AhrsQuboNode();

	void update();

//	void ahrsCallBack(const sensor_msgs::Imu msg)

protected:

	std::string name;

	//data retrieved from the sensor
	AHRS::AHRSData sensor_data;
	sensor_msgs::Imu msg;

	//determines how many times we should try to connect to the sensor
	const int MAX_CONNECTION_ATTEMPTS = 10;
	bool isConnected = false;

	ros::Publisher ahrsPub;

	std::unique_ptr<AHRS> ahrs;

};

#endif
