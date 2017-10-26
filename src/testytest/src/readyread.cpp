#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

void pieCallback(const std_msgs::String::ConstPtr& msg)
{
	ROS_INFO("type of pie:[%s]", msg->data.c_str());
}

int main (int argc, char **argv)
{
	ros::init(argc, argv, "readyread");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("pies", 1000, pieCallback);
	ros::spin();
	
	return 0;
}
