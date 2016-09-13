#include "ahrs_qubo.h"
//written by Jeremy Weed

/* Main file for the Qubo AHRS node */

#define AHRS_0_FILE "/currently/unknown"

int main(int argc, char ** argv){

	if (argc != 4){
		ROS_ERROR("The AHRS node received %i arguments which is not correct\n",
		argc);
		exit(1);
	}

	//needs to be called first
	ros::init(argc, argv, "ahrs_node");
	std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);

	//the pointer to the ahrs node
	std::unique_ptr<RamNode> node0;

	//sets the pointer up
	node0.reset(new AhrsQuboNode(n, 10, "AHRS_0", AHRS_0_FILE));

	while (ros::ok()){
		node0->update();
	}
}
