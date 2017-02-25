#ifndef DVL_QUBO_HEADER
#define DVL_QUBO_HEADER
//written by Jeremy Weed

#include "qubo_node.h"
#include "ram_msgs/DVL_qubo.h"
#include "DVL.h"

class DvlQuboNode : public QuboNode {

public:

	/**
	 * Constructor for the DVL ROS node
	 * @param n      shared_ptr to the NodeHandle that this node will publish on
	 * @param rate   rate handed to ros::rate
	 * @param name   device description, used when publishing data
	 * @param device file location of the DVL device
	 */
	DvlQuboNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string name,
	std::string device);

	/**
	 * Desctructor for the DVL ROS node
	 */
	~DvlQuboNode();

	void update();

protected:
	/**
	 * identifier for the DVL
	 */
	std::string name;

	/**
	 * ROS publisher for the DVL data
	 */
	ros::Publisher dvlPub;

	/**
	 * unique_ptr to the DVL
	 */
	std::unique_ptr<DVL> dvl;

	/**
	 * describes how many times we should try to reconnect to the device before
	 * just killing the node
	 */
	const int MAX_CONNECTION_ATTEMPTS = 10;

	/**
	 * defined in DVL driver
	 * its the data the DVL collects
	 */
	DVL::DVLData sensor_data;

	/**
	 * defined in DVL types
	 * data used to get more accurate readings from the DVL
	 */
	DVL::LiveConditions live_cond;

	/**
	 * defined in DVL types
	 * sets up the DVL
	 */
	DVL::VehicleConfig v_config;

	/**
	 *
	 */
	void setup_dvl();
};
#endif
