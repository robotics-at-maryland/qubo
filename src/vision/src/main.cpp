#include "vision_node.h"

typedef actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> Server;

int main(int argc, char** argv){

	//TODO finalize how we handle this
	if(argc < 4){
        ROS_ERROR("wrong number of arguments passed to vision node! you passed in %i, we wanted 4. node will exit now", argc);
		ROS_ERROR("for reference your argument:");
		for(int i = 0; i < argc; i++){
			ROS_ERROR("%i = %s", i, argv[i]);
		}
		
		exit(0); 
    }
    // init the node handle
    ros::init(argc,argv, "vision_node");
    
    //make a pointer to a node handle, I'm actually not even sure we need the node handle...
    ros::NodeHandle n;
    VisionNode node(n,argv[1]);
    
    ros::Rate r(10); //not 10 hz
    while(ros::ok()){
		//		ROS_ERROR("about to call update");
        node.update();
		//		ROS_ERROR("in main");
		// r.sleep(); //you update this time in the second argument to the VisionNode constructor
    }

    return 0;
}

