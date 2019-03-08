#ifndef QSCU_NODE
#define QSCU_NODE

// ros
#include "ros/ros.h"
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
	};

	std::string m_node_name;

	QSCU qscu;

	ros::Timer qubobus_loop;
	ros::Timer qubobus_incoming_loop;
	std::queue<QMsg> m_outgoing;
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

	ros::Subscriber m_thruster_zero;
	void thrusterZeroCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_one;
	void thrusterOneCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_two;
	void thrusterTwoCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_three;
	void thrusterThreeCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_four;
	void thrusterFourCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_five;
	void thrusterFiveCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_six;
	void thrusterSixCallback(const std_msgs::Float64::ConstPtr& msg);

	ros::Subscriber m_thruster_seven;
	void thrusterSevenCallback(const std_msgs::Float64::ConstPtr& msg);

        

  // The callbacks will write into the buffer, and the actual commands get these buffer's value
  // only after a loop of qscu's thruster message being sent
	float m_thruster_zero_buffer = 0;
	float m_thruster_one_buffer = 0;
	float m_thruster_two_buffer = 0;
	float m_thruster_three_buffer = 0;
	float m_thruster_four_buffer = 0;
	float m_thruster_five_buffer = 0;
	float m_thruster_six_buffer = 0;
	float m_thruster_seven_buffer= 0;

	// We need these so we can store the last message that was sent on the bus
	// to calculate the value we need to give the thrusters
	//float m_yaw_command		= 0;
	//float m_pitch_command	= 0;
	//float m_roll_command	= 0;
	//float m_depth_command	= 0;
	//float m_surge_command	= 0;
	//float m_sway_command	= 0;
	bool thruster_update    = false;

	ros::Timer qubobus_thruster_loop;
	void QubobusThrusterCallback(const ros::TimerEvent&);

	ros::Timer qubobus_status_loop;
	void QubobusStatusCallback(const ros::TimerEvent&);
};

#endif
