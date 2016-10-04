#include "ahrs_qubo.h"
//written by Jeremy Weed


int main(int argc, char ** argv){

	if (argc != 4){
		ROS_ERROR("The AHRS node received %i arguments which is not correct"
		" (should be 4)\n", argc);
		for(int i = 0; i < argc; i++){
			ROS_ERROR("found: %s", argv[i]);
		}
		exit(1);
	}
	ROS_DEBUG("Made it past the argument checks");
	//needs to be called first
	ros::init(argc, argv, "ahrs_node");
	std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);

	//the pointer to the ahrs node
	std::unique_ptr<RamNode> node0;

	//sets the pointer up
	node0.reset(new AhrsQuboNode(n, 10, "AHRS_0", argv[1]));

	while (ros::ok()){
		node0->update();
	}
}
