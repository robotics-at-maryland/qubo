#include "ahrs_qubo.h"
//written by Jeremy Weed

/**
* main method for the AHRS nodes
* handles multiple devices across the robot
* and updates them at a specified rate
* @param  argc number of cmd-line arguments found - its a c/c++ thing
* @param  argv char* array of the cmd-line arguments
*              these are handed to us by the launch file
*/
int main(int argc, char ** argv){

	//Currently, we expect 4 arguments:
	//	name, pkg, type, and one actual argument containing the file
	//	location of the AHRS device
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

	//create a new shared_ptr to the NodeHandle
	std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);

	//the pointer to the ahrs node
	std::unique_ptr<QuboNode> node0;

	//sets the pointer up
	//n is the NodeHandle
	//10 is the loop_rate
	//"AHRS_0" is the name of the device - also the locations of published messages
	//the file location of the AHRS is argument 1
	node0.reset(new AhrsQuboNode(n, 10, "AHRS_0", argv[1]));

	//read and sleep
	while (ros::ok()){
		node0->update();
		node0->sleep();
	}
}
