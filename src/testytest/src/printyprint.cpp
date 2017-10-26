#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

int main (int argc, char **argv) {
	//initialize and set up node
	ros::init(argc, argv, "printyprint");
	ros::NodeHandle n;

	//create the publisher
	ros::Publisher pie_pub = n.advertise<std_msgs::String>("pies",1000);

	//frequency
	ros::Rate loop_rate(10);

	//count how many messages sent
	int count = 0;
	
	//the message!
	while (ros::ok())
	{


		std_msgs::String msg;
		std::stringstream ss;
		ss << " apple pie " << count;
		msg.data = ss.str();

		ROS_INFO("%s", msg.data.c_str());

		pie_pub.publish(msg);

		ros::spinOnce();

		loop_rate.sleep();		
		++count;
	}

	return 0;
}



