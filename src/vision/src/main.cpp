#include "vision_node.h"

int main(int argc, char** argv){

	//TODO finalize how we handle this
	if(argc < 2){
        ROS_ERROR("wrong number of arguments passed to vision node! you passed in %i, we need at least 2", argc);
		ROS_ERROR("for reference your argument:");
		for(int i = 0; i < argc; i++){
			ROS_ERROR("%i = %s", i, argv[i]);
		}
		
		exit(0); 
    }
    // init the node handle
    ros::init(argc,argv, "vision_node");
	ros::NodeHandle n;
	ros::NodeHandle np("~");
	
    VisionNode node(n,np,argv[1]); //argv[1] should be the topic the camera is published on

	ros::Rate r(10); //not 10 hz
    while(ros::ok()){
        node.update();
		r.sleep(); //you update this time in the second argument to the VisionNode constructor
    }

    return 0;
}

