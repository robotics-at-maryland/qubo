#include "ros/ros.h"
#include "std_msgs/String.h"

int main(int argc, char** argv) {
	ros::init(argc, argv, "noob_pub");
	ros::NodeHandle n;
	ros::Publisher pub = n.advertise<std_msgs::String>("nooby_info", 1000);
	
	ros::Rate loop_rate(10);
	std_msgs::String msg;

	while(ros::ok()) {
		msg.data = "hello world";	

		pub.publish(msg);
		loop_rate.sleep();
	}

	return 0;
}
