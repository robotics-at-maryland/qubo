#include "gpu_vision_node.h"


int main(int argc, char** argv){

	if(argc != 6){
		ROS_ERROR("wrong number of arguments passed to vision node!"
			"You passed %i, and we wanted 6.  Node will exit now.", argc);
		ROS_ERROR("We've standardized around using launch files to launch our "
			"nodes, yours should have the follwing line:\n"
			"<node name=\"vision_node\" pkg=\"vision\" type=\"vision_node\" "
			"args=\"feed0 feed1 feedb\"/>\nwhere feed0 and feed1 and feedb are"
			"either physical camera paths. \n See the roslaunch folder ing the "
			"drive if you want to find out what the other arguments roslaunch "
			"uses are.");
		exit(0);
	}
	ros::init(argc, argv,"gpu_vision_node");

	std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
	GpuVisionNode node(n, argv[1], argv[2], argv[3]);

	ros::Rate r(10);  //10 Hz
	while(ros::ok()){
		node.update();
		r.sleep();
	}

	return 0;
}
