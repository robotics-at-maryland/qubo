#ifndef MOVE_CORE_H
#define MOVE_CORE_H

// ROS includes.
#include "tortuga_node.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Int64MultiArray.h"
//#include "tortuga/sensorapi.h"


// Custom message includes. Auto-generated from msg/ directory.
#include <sensor_msgs/Joy.h>

class moveNode : public TortugaNode {

	public:
		//! Constructor.
  		moveNode(int, char** , int);
		
		void update();
		void publish();
  		//! Destructor.
  		~moveNode();

	private:

  		//! Callback function for subscriber.
  		void messageCallback(const std_msgs::Float64MultiArray::ConstPtr &msg);

		ros::NodeHandle nh;
		ros::Subscriber joystick_sub;
		ros::Publisher thrust_pub;
		int thrstr_1_spd, thrstr_2_spd, thrstr_3_spd, thrstr_4_spd, thrstr_5_spd, thrstr_6_spd;
};

#endif // MOVE_CORE_H

