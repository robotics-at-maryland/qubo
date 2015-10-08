#include "ros/ros.h"
#include <sstream>

int main(int argc, char **argv){

	ros::init(argc, argv, "test");

	ros::NodeHandle n;

	while(ros::ok()){
		printf("hello world");
	}
}