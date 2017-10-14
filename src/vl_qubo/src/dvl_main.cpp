#include "dvl_qubo.h"

//written by Jeremy Weed

/**
 * main method for the DVL node
 * Updates data from the DVL at a specified rate
 * @param  argc number of cmd-line arguments found
 * @param  argv char* array oof the cmd-line arguments
 * @return      an exit code
 */
int main(int argc, char** argv){

	if (argc != 4){
		ROS_ERROR("The DVL node recieved %i arguments, which is not correct"
		" (should be 4)\n", argc);
		for(int i = 0; i < argc; i++){
			ROS_ERROR("found: %s", argv[i]);
		}
		exit(1);
	}
	//initializes the node
	ros::init(argc, argv, "dvl_node");

	//create a shared_ptr to the NodeHandle
	ros::NodeHandle nh;

	//pointer to the node
	DvlQuboNode node0(nh, 10, "DVL", argv[1]);

	
	//read and sleep
	while(ros::ok()){
		node0.update();
		
	}
}
