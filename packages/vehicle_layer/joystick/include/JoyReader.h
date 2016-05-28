#ifndef JOY_CORE_H
#define JOY_CORE_H

#include "ros/ros.h"
#include "tortuga_node.h"
#include "sensor_msgs/Joy.h"
#include "std_msgs/Float64MultiArray.h"


class JoyReader : public TortugaNode {

 	public:
  		JoyReader(int, char**, int);
  		~JoyReader();
		
		void update();
		void publish();

	private:
//		ros::Rate loop_rate;
  		ros::NodeHandle n;
		float x, y, z, mag;
  		ros::Subscriber subscriber;
  		ros::Publisher publisher;
  		void joyPub(const sensor_msgs::Joy::ConstPtr &);

};

#endif
