#include "ros/ros.h"
#include "std_msgs/String.h"

void callback(const std_msgs::String::ConstPtr& msg); 

int main(int argc, char** argv) {
	ros::init(argc, argv, "noob_sub");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("nooby_info", 1000, callback);
	
	ros::spin();

	return 0;
}

void callback(const std_msgs::String::ConstPtr& msg) {
	ROS_INFO("Recieved message: %s", msg->data.c_str());
}
