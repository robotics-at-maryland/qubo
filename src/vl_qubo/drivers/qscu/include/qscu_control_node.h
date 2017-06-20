#ifndef QSCU_CONTROL_NODE
#define QSCU_CONTROL_NODE

// ros
#include "ros/ros.h"
#include "sensor_msgs/Imu.h" // Probably don't need this one
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"

// c++ stuff
#include <iostream>
#include <sstream>
#include <thread>
#include <stdio.h>

// No idea what this is
#include "tf/tf.h"

class QSCUControlNode {

	public:
	QSCUControlNode(ros::NodeHandle n, std::string node_name, std::string pose);
	~QSCUControlNode();

	void update();

	protected:
	std::string m_node_name;

	
};

#endif
