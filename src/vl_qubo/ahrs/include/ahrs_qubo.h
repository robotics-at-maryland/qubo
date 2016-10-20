#ifndef AHRS_QUBO_HEADER
#define AHRS_QUBO_HEADER
//written by Jeremy Weed

#include "qubo_node.h"
#include "AHRS.h"
#include "sensor_msgs/Imu.h"



class AhrsQuboNode : public QuboNode {

public:

	/**
	 * Constructor for the AHRS ROS node
	 * n		shared_ptr 	the NodeHandle that this node will publish on
	 * rate		int 		describes the rate handed to ros::rate
	 * name		String	 	describes this device, used when publishing data
	 * device	String 		File location of the ahrs device
	 */
	AhrsQuboNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string name,
		std::string device);

	/**
	 * Desctructor for the AHRS ROS node
	 */
	~AhrsQuboNode();

	/**
	 * Update method for the AHRS ROS node
	 * call this to read data from the device, and then immediately publish
	 * what was found
	 */
	void update();


protected:

	/**
	 * The identifier for this specific AHRS
	 * each AHRS on the robot should have a unique name,
	 * because its used to distinguish the messages published by each device
	 */
	std::string name;

	/**
	 * defined in the AHRS driver
	 * Describes the data the sensor collects
	 */
	AHRS::AHRSData sensor_data;

	/**
	 * ROS default message for IMU devices
	 */
	sensor_msgs::Imu msg;

	/**
	 * describes how many times we should try to reconnect to the device before
	 * just killing the node
	 */
	const int MAX_CONNECTION_ATTEMPTS = 10;

	/**
	 *	ROS publisher for the sensor_msgs::Imu data collected from the AHRS
	 */
	ros::Publisher ahrsPub;

	/**
	 * unique_ptr to this AHRS node
	 */
	std::unique_ptr<AHRS> ahrs;

};

#endif
