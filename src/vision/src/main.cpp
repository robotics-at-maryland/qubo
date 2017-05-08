#include "vision_node.h"

typedef actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> Server;

int main(int argc, char** argv){
    if(argc != 4){
        ROS_ERROR("wrong number of arguments passed to vision node! you passed in %i, we wanted 4. node will exit now", argc);
        ROS_ERROR("we've standardized around using launch files to launch our nodes, yours should have the following line\n<node name=\"vision_node\" pkg=\"vision\" type=\"vision_node\" args=\"feed\"/>\n where feed is either physical camera paths .\n See the roslaunch folder in the drive if you want to find out what the other arguments roslaunch uses are"); 
        exit(0); 
    }
    // init the node handle
    ros::init(argc,argv, "vision_node");
    
    //make a pointer to a node handle, I'm actually not even sure we need the node handle...
    std::shared_ptr<ros::NodeHandle> n(new ros::NodeHandle);
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

