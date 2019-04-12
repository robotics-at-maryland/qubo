#ifndef QSCU_NODE
#define QSCU_NODE

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
#include <boost/any.hpp>
#include <boost/variant.hpp>
#include <boost/optional.hpp>
#include <queue>
#include <utility>

// No idea what this is
#include "tf/tf.h"

#include "QSCU.h"
#include "qubobus.h"
#include "io.h"


class QSCUNode {

	public:
	QSCUNode(ros::NodeHandle n, std::string node_name, std::string pose);
	~QSCUNode();

	void update();

	protected:

	// This is a class to hold messages passed in a queue to the timer which controls the bus
	// I didn't want to just pass void* around like we do on the Tiva, so I'm using shared ptrs
	class QMsg {
		public:

		// By specifying Error first, variant::which() will return 0 for an error
		Transaction type;
		std::shared_ptr<void> payload;
		std::shared_ptr<void> reply;
		const bool operator<(const QMsg& obj) const;
	};

	std::string m_node_name;

	QSCU qscu;

	ros::Timer qubobus_loop;
	ros::Timer qubobus_incoming_loop;

	const int	THRUSTER_PRIORITY = 127;
	const int	STATUS_PRIORITY	  = 0;

	// bool QMsgCompare(std::pair<int, QMsg> p1, std::pair<int, QMsg> p2);
	// , std::vector<std::pair<int, QMsg>, decltype(QMsgCompare)>
	std::priority_queue<std::pair<int, QMsg>> m_outgoing;
	std::queue<QMsg> m_incoming;

	void QubobusCallback(const ros::TimerEvent&);
	void QubobusIncomingCallback(const ros::TimerEvent&);

	/**************************************************************
	 * Publishers and the messages they use                       *
	 **************************************************************/
	ros::Publisher m_status_pub;
	ram_msgs::Status m_status_msg;

	/**************************************************************
	 * Subscribers, their callbacks, and the messages they use    *
	 **************************************************************/
	std::vector<float> m_thruster_speeds;

	ros::Subscriber m_yaw_sub;
	void yawCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_pitch_sub;
	void pitchCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_roll_sub;
	void rollCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_depth_sub;
	void depthCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_surge_sub;
	void surgeCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_sway_sub;
	void swayCallback(const std_msgs::Float64::ConstPtr& msg);

  // The callbacks will write into the buffer, and the actual commands get these buffer's value
  // only after a loop of qscu's thruster message being sent
	float m_yaw_command_buffer = 0;
	float m_pitch_command_buffer = 0;
	float m_roll_command_buffer = 0;
	float m_depth_command_buffer = 0;
	float m_surge_command_buffer = 0;
	float m_sway_command_buffer = 0;

	// We need these so we can store the last message that was sent on the bus
	// to calculate the value we need to give the thrusters
	float m_yaw_command		= 0;
	float m_pitch_command	= 0;
	float m_roll_command	= 0;
	float m_depth_command	= 0;
	float m_surge_command	= 0;
	float m_sway_command	= 0;
	bool thruster_update    = false;

	ros::Timer qubobus_thruster_loop;
	void QubobusThrusterCallback(const ros::TimerEvent&);

	ros::Timer qubobus_status_loop;
	void QubobusStatusCallback(const ros::TimerEvent&);
};

#endif
