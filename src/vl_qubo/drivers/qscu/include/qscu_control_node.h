#ifndef QSCU_CONTROL_NODE
#define QSCU_CONTROL_NODE

// ros
#include "ros/ros.h"
#include "sensor_msgs/Imu.h" // Probably don't need this one
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"

// Custom messages
#include "ram_msgs/Status.h"

// c++ stuff
#include <iostream>
#include <sstream>
#include <thread>
#include <stdio.h>

// No idea what this is
#include "tf/tf.h"

#include "QSCU.h"

class QSCUControlNode {

	public:
	QSCUControlNode(ros::NodeHandle n, std::string node_name, std::string pose);
	~QSCUControlNode();

	void update();

	protected:
	std::string m_node_name;

	QSCU qscu;
	ros::Publisher m_status_pub;
	ram_msgs::Status m_status_msg;

	ros::Timer qubobus_loop;
	void QubobusCallback(const ros::TimerEvent&);
};

#endif
